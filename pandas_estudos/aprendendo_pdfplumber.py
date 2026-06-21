import re
import calendar
from datetime import date
from pathlib import Path

import pandas as pd
import pdfplumber

MESES_PT = {
    "janeiro": "01", "fevereiro": "02", "março": "03", "abril": "04",
    "maio": "05", "junho": "06", "julho": "07", "agosto": "08",
    "setembro": "09", "outubro": "10", "novembro": "11", "dezembro": "12",
}


def parse_br_number(value: str) -> float:
    return float(value.strip().replace(".", "").replace(",", "."))


def _extrair_texto(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _extrair_mes_referencia(texto: str) -> str:
    padrao = r"(" + "|".join(MESES_PT) + r")[/\s]+(\d{4})"
    m = re.search(padrao, texto, re.IGNORECASE)
    if not m:
        raise ValueError("Mês de referência não encontrado no PDF.")
    mes = MESES_PT[m.group(1).lower()]
    return f"{m.group(2)}-{mes}"


def _sinal_cashflow(descricao: str) -> int:
    """Resgates são saídas (−); aportes e aplicações são entradas (+)."""
    return -1 if "RESGATE" in descricao.upper() else +1


def _cf_date(data_str: str, mes_ref: str) -> date:
    """Infere o ano do fluxo a partir do DD/MM e do mês de referência."""
    dia, mes_mov = int(data_str[:2]), int(data_str[3:5])
    ano_ref, mes_ref_num = int(mes_ref[:4]), int(mes_ref[5:])
    ano = ano_ref - 1 if mes_mov > mes_ref_num else ano_ref
    return date(ano, mes_mov, dia)


def _peso_md(cf_date: date, mes_ref: str) -> float:
    """W_i = (CD − D_i) / CD, onde D_i = dias desde o início do período."""
    ano, mes = int(mes_ref[:4]), int(mes_ref[5:])
    cd = calendar.monthrange(ano, mes)[1]
    d_i = (cf_date - date(ano, mes, 1)).days
    return (cd - d_i) / cd


def _modified_dietz(
    bmv: float,
    emv: float,
    cfs: list[tuple[date, float]],
    mes_ref: str,
) -> float | None:
    """
    R = (EMV − BMV − ΣCF) / (BMV + Σ(CF_i × W_i))
    Retorna None se não houver capital alocado no período.
    """
    cf_total = sum(v for _, v in cfs)
    denominador = bmv + sum(v * _peso_md(d, mes_ref) for d, v in cfs)
    if denominador == 0:
        return None
    return (emv - bmv - cf_total) / denominador


# Captura linhas de movimentação: DD/MM + descrição (exceto SALDO) + 4 valores
_CF_RE = re.compile(
    r"^(\d{2}/\d{2})\s+(?!SALDO)([A-ZÁÉÍÓÚÃÕÇ][^\d\n]*?)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)",
    re.MULTILINE,
)


def _extrair_cashflows(
    bloco: str,
    mes_ref: str,
    bmv: float,
    emv: float,
    data_aplicacao: str | None,
) -> list[tuple[date, float]]:
    """
    Extrai fluxos de caixa do bloco da aplicação.
    Para novas aplicações (BMV=0 sem entrada explícita), infere o aporte
    pela data_aplicacao e pelo saldo atual.
    """
    cfs = []
    for m in _CF_RE.finditer(bloco):
        data_str, descricao, _, valor_bruto_str, _, _ = m.groups()
        valor = parse_br_number(valor_bruto_str) * _sinal_cashflow(descricao)
        cfs.append((_cf_date(data_str, mes_ref), valor))

    entrada_ja_presente = any(v > 0 for _, v in cfs)
    if bmv == 0 and emv > 0 and not entrada_ja_presente and data_aplicacao:
        dia = int(data_aplicacao[:2])
        mes_ap = int(data_aplicacao[3:5])
        ano_ap = 2000 + int(data_aplicacao[6:])
        cfs.append((date(ano_ap, mes_ap, dia), emv))

    return cfs


def _processar_pdf(pdf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lê um extrato PDF e retorna:
      - df_inv: uma linha por aplicação (com retorno Modified Dietz)
      - df_cf:  uma linha por fluxo de caixa
    """
    texto = _extrair_texto(pdf_path)
    mes_ref = _extrair_mes_referencia(texto)
    blocos = re.split(r"Aplica[çc][aã]o N[°º]\s+", texto)

    registros_inv, registros_cf = [], []

    for bloco in blocos[1:]:
        num_match = re.match(r"(\d+)", bloco)
        if not num_match:
            continue
        num = num_match.group(1)

        prod_match = re.search(
            r"^(.+?)\s+(\d{2}/\d{2}/\d{2})\s+([\d.,]+)\s+([\d.,]+)\s+(\d{2}/\d{2}/\d{2})",
            bloco, re.MULTILINE,
        )
        saldo_ant = re.search(
            r"SALDO ANTERIOR[¹]?\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)", bloco
        )
        saldo_at = re.search(
            r"SALDO ATUAL[¹]?\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)", bloco
        )
        rend_match = re.search(
            r"Rendimento Bruto no Per[ií]odo \(R\$\)\s+([\d.,]+)", bloco
        )

        bmv = parse_br_number(saldo_ant.group(2)) if saldo_ant else 0.0
        emv = parse_br_number(saldo_at.group(2)) if saldo_at else 0.0
        data_ap = prod_match.group(2) if prod_match else None

        cfs = _extrair_cashflows(bloco, mes_ref, bmv, emv, data_ap)
        retorno_md = _modified_dietz(bmv, emv, cfs, mes_ref)

        registros_inv.append({
            "mes_referencia": mes_ref,
            "num_aplicacao": num,
            "produto": prod_match.group(1).strip() if prod_match else None,
            "pct_indexador": parse_br_number(prod_match.group(4)) if prod_match else None,
            "data_vencimento": prod_match.group(5) if prod_match else None,
            "saldo_bruto_anterior": bmv,
            "saldo_bruto": emv,
            "saldo_liquido": parse_br_number(saldo_at.group(4)) if saldo_at else None,
            "rendimento_bruto_periodo": parse_br_number(rend_match.group(1)) if rend_match else None,
            "retorno_md": retorno_md,
        })

        for cf_date, cf_valor in cfs:
            registros_cf.append({
                "mes_referencia": mes_ref,
                "num_aplicacao": num,
                "data": cf_date,
                "valor": cf_valor,
                "tipo": "SAIDA" if cf_valor < 0 else "ENTRADA",
            })

    return pd.DataFrame(registros_inv), pd.DataFrame(registros_cf)


def _retorno_portfolio(df_inv: pd.DataFrame, df_cf: pd.DataFrame) -> pd.DataFrame:
    """Modified Dietz consolidado do portfólio, agrupado por mês."""
    resultados = []
    for mes_ref in sorted(df_inv["mes_referencia"].unique()):
        grp_inv = df_inv[df_inv["mes_referencia"] == mes_ref]
        grp_cf = df_cf[df_cf["mes_referencia"] == mes_ref]

        bmv = grp_inv["saldo_bruto_anterior"].sum()
        emv = grp_inv["saldo_bruto"].sum()
        cfs = list(zip(grp_cf["data"], grp_cf["valor"]))

        resultados.append({
            "mes_referencia": mes_ref,
            "saldo_bruto_anterior": bmv,
            "saldo_bruto": emv,
            "saldo_liquido": grp_inv["saldo_liquido"].sum(),
            "rendimento_bruto": grp_inv["rendimento_bruto_periodo"].sum(),
            "retorno_md": _modified_dietz(bmv, emv, cfs, mes_ref),
        })

    return pd.DataFrame(resultados)


def processar_extrato(pdf_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processa um único extrato PDF.
    Retorna (df_por_aplicacao, df_por_mes).
    """
    df_inv, df_cf = _processar_pdf(pdf_path)
    return df_inv, _retorno_portfolio(df_inv, df_cf)


def processar_pasta(pasta: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processa todos os extratos PDF de uma pasta, em ordem cronológica.
    Retorna (df_por_aplicacao, df_por_mes).
    """
    pdfs = sorted(pasta.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {pasta}")
    resultados = [_processar_pdf(p) for p in pdfs]
    df_inv = pd.concat([r[0] for r in resultados], ignore_index=True)
    df_cf = pd.concat([r[1] for r in resultados], ignore_index=True)
    return df_inv, _retorno_portfolio(df_inv, df_cf)


def exportar_excel(
    df_aplicacoes: pd.DataFrame,
    df_portfolio: pd.DataFrame,
    destino: Path,
) -> None:
    """Exporta os dois DataFrames para abas separadas de um arquivo Excel."""
    percent_fmt = "0.00%"
    reais_fmt = 'R$ #,##0.00'

    colunas_percent_ap = ["pct_indexador", "retorno_md"]
    colunas_reais_ap = [
        "saldo_bruto_anterior", "saldo_bruto", "saldo_liquido", "rendimento_bruto_periodo"
    ]
    colunas_percent_port = ["retorno_md"]
    colunas_reais_port = [
        "saldo_bruto_anterior", "saldo_bruto", "saldo_liquido", "rendimento_bruto"
    ]

    with pd.ExcelWriter(destino, engine="openpyxl") as writer:
        df_aplicacoes.to_excel(writer, sheet_name="Por Aplicação", index=False)
        df_portfolio.to_excel(writer, sheet_name="Portfolio por Mês", index=False)

        wb = writer.book

        for sheet_name, colunas_r, colunas_p, df in [
            ("Por Aplicação", colunas_reais_ap, colunas_percent_ap, df_aplicacoes),
            ("Portfolio por Mês", colunas_reais_port, colunas_percent_port, df_portfolio),
        ]:
            ws = wb[sheet_name]

            # Largura automática das colunas
            for col_cells in ws.columns:
                max_len = max(len(str(c.value or "")) for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = max_len + 4

            # Formata colunas de reais e percentual
            col_idx = {c: i + 1 for i, c in enumerate(df.columns)}
            for col_name, fmt in (
                [(c, reais_fmt) for c in colunas_r] + [(c, percent_fmt) for c in colunas_p]
            ):
                if col_name not in col_idx:
                    continue
                col_letter = ws.cell(1, col_idx[col_name]).column_letter
                for row in ws.iter_rows(min_row=2, min_col=col_idx[col_name], max_col=col_idx[col_name]):
                    for cell in row:
                        cell.number_format = fmt

    print(f"Exportado para: {destino.resolve()}")


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Caso 1: PDF único
    df_aplicacoes, df_portfolio = processar_extrato(Path("data/extrato_sem_infos.pdf"))

    # Caso 2: pasta com múltiplos PDFs
    # df_aplicacoes, df_portfolio = processar_pasta(Path("data/extratos/"))

    print("=== Por Aplicação ===")
    print(df_aplicacoes)
    print("\n=== Portfolio por Mês (Modified Dietz) ===")
    print(df_portfolio)

    exportar_excel(df_aplicacoes, df_portfolio, Path("data/consolidado_portfolio.xlsx"))

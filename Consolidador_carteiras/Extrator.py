"""
Extrator de Extrato Consolidado Santander
==========================================
Extrai e estrutura:
  1. Resumo (saldos e totais)
  2. Movimentação da Conta Corrente
  3. CDB/RDB — Movimentação por aplicação

Uso:
    python extrator_santander.py Extrato_consolidado_mensal.pdf
"""
# %%

import re, sys
import pdfplumber
import pandas as pd
from pathlib import Path

VALOR_RE = re.compile(r"^\d{1,3}(?:\.\d{3})*,\d{2}-?$")
DATA_RE  = re.compile(r"^\d{2}/\d{2}$")
TIPOS_TX = re.compile(
    r"^(DEBITO|PIX|TED|DOC|RESGATE|APLICACAO|PAGAMENTO|REMUNERACAO|"
    r"MENSALIDADE|SAQUE|DEPOSITO|TRANSFERENCIA|IOF|LIQUIDO|DEBITO AUT)",
    re.IGNORECASE
)

def parse_valor(s):
    s = s.strip()
    neg = s.endswith("-")
    s = s.rstrip("-").replace(".", "").replace(",", ".")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None

def eh_valor(tok):
    return bool(VALOR_RE.match(tok.strip()))

def limpa_linhas(texto):
    ignorar = {
        "Extrato_PF_A4_Inteligente - 27/11/2024",
        "BALP_UY_M3SM4504_MXDD0226.PIM -",
        "EXTRATO CONSOLIDADO INTELIGENTE",
        "fevereiro/2026",
        "Data Descrição Nº Documento Movimento (R$) Saldo (R$)",
    }
    out = []
    for linha in texto.split("\n"):
        linha = linha.strip()
        if not linha: continue
        if re.match(r"P[áa]gina:", linha, re.I): continue
        if linha in ignorar: continue
        out.append(linha)
    return out

# ── Resumo ────────────────────────────────────────────────────────────────
def extrair_resumo(linhas):
    dentro, recs = False, []
    for linha in linhas:
        if "Resumo - fevereiro/2026" in linha:
            dentro = True; continue
        if dentro and linha.strip() == "Conta Corrente": break
        if not dentro: continue
        m = re.match(r"^(.*?)\s+([\d.,]+-?)$", linha)
        if m:
            v = parse_valor(m.group(2))
            if v is not None:
                recs.append({"Descrição": m.group(1).strip(), "Valor (R$)": v})
    return pd.DataFrame(recs)

# ── Movimentação Conta Corrente ────────────────────────────────────────────
def extrair_movimentacao(linhas):
    # 1) Delimita o bloco
    dentro, bloco = False, []
    for linha in linhas:
        if "SALDO EM 31/01" in linha: dentro = True
        if dentro: bloco.append(linha)
        if dentro and "SALDO EM 28/02" in linha: break

    # 2) Uma linha inicia transação quando tem valor no final OU começa com data/tipo
    def eh_inicio_tx(linha):
        t = linha.split()
        if not t: return False
        if linha.startswith("SALDO"): return True
        if DATA_RE.match(t[0]): return True
        if TIPOS_TX.match(linha): return True
        if eh_valor(t[-1]): return True
        return False

    # 3) Agrupa linhas
    grupos = []
    for linha in bloco:
        if eh_inicio_tx(linha):
            grupos.append([linha])
        elif grupos:
            grupos[-1].append(linha)

    # 4) Parse de cada grupo
    recs = []
    data_atual = None

    for grupo in grupos:
        principal = grupo[0]
        detalhes = []
        for g in grupo[1:]:
            gt = g.split()
            if not gt: continue
            if DATA_RE.match(gt[0]):
                detalhes.append(" ".join(gt[1:]))
            elif not TIPOS_TX.match(g) and not eh_valor(gt[-1]):
                detalhes.append(g)

        tokens = principal.split()
        if not tokens: continue

        if principal.startswith("SALDO"):
            parts = principal.rsplit(None, 1)
            v = parse_valor(parts[-1]) if len(parts) == 2 else None
            recs.append({"Data": None, "Descrição": parts[0],
                         "Beneficiário/Detalhe": "", "Nº Documento": "",
                         "Movimento (R$)": v, "Saldo (R$)": v})
            continue

        offset = 0
        if DATA_RE.match(tokens[0]):
            data_atual = tokens[0]; offset = 1

        fim = len(tokens)
        saldo = movimento = None
        if fim > offset and eh_valor(tokens[fim-1]):
            if fim-1 > offset and eh_valor(tokens[fim-2]):
                saldo     = parse_valor(tokens[fim-1])
                movimento = parse_valor(tokens[fim-2])
                fim -= 2
            else:
                movimento = parse_valor(tokens[fim-1])
                fim -= 1

        num_doc = ""
        if fim > offset and re.match(r"^\d{6}$", tokens[fim-1]):
            num_doc = tokens[fim-1]; fim -= 1
        if fim > offset and tokens[fim-1] == "-":
            fim -= 1

        descricao = " ".join(tokens[offset:fim])
        beneficiario = " | ".join(d for d in detalhes if d.strip())

        if movimento is None and not descricao: continue
        recs.append({"Data": data_atual, "Descrição": descricao,
                     "Beneficiário/Detalhe": beneficiario,
                     "Nº Documento": num_doc,
                     "Movimento (R$)": movimento, "Saldo (R$)": saldo})

    df = pd.DataFrame(recs)
    df["Saldo (R$)"] = df["Saldo (R$)"].ffill()

    # 5) Pós-processamento: merge de linhas de estabelecimento soltas
    #    Padrão: linha sem movimento logo após "DEBITO VISA ELECTRON BRASIL"
    #    → o nome do estabelecimento vai para Beneficiário/Detalhe da linha anterior
    to_drop = []
    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        if (pd.isna(row["Movimento (R$)"]) and
                not str(row["Descrição"]).startswith("SALDO") and
                "DEBITO VISA ELECTRON BRASIL" in str(prev["Descrição"])):
            df.at[df.index[i - 1], "Beneficiário/Detalhe"] = row["Descrição"]
            to_drop.append(df.index[i])

    df = df.drop(index=to_drop).reset_index(drop=True)
    return df

# ── CDB/RDB ────────────────────────────────────────────────────────────────
def extrair_cdb(linhas):
    VALOR_MOV = re.compile(r"^\d{1,3}(?:\.\d{3})*,\d{2}$")
    IGNORAR_CDB = {
        "Data Descrição Valor Principal (R$) Valor Bruto (R$) Valor IR/IOF (R$) Valor Líquido (R$)",
        "Movimentação", "2", "¹",
    }

    idx_aplic = []
    for i, linha in enumerate(linhas):
        m = re.match(r"Aplica[çc][aã]o N[°º]\s+(\d+)", linha)
        if m:
            idx_aplic.append((m.group(1), i))

    recs = []
    for k, (num, idx) in enumerate(idx_aplic):
        fim_bloco = idx_aplic[k+1][1] if k+1 < len(idx_aplic) else len(linhas)
        bloco = linhas[idx:fim_bloco]

        em_mov = False
        for linha in bloco:
            if "Valor Principal (R$)" in linha:
                em_mov = True; continue
            if not em_mov: continue
            if linha in IGNORAR_CDB: continue
            if linha.startswith("¹ Saldo") or linha.startswith("Rendimento"): break

            tokens = linha.split()
            if len(tokens) < 2: continue
            if not DATA_RE.match(tokens[0]): continue

            data   = tokens[0]
            vals   = [t for t in tokens[1:] if VALOR_MOV.match(t)]
            desc_t = [t for t in tokens[1:] if not VALOR_MOV.match(t)]
            desc   = " ".join(desc_t)
            parsed = [parse_valor(v) for v in vals]
            while len(parsed) < 4: parsed.append(None)

            recs.append({"Aplicação N°": num, "Data": data, "Descrição": desc,
                         "Valor Principal (R$)": parsed[0],
                         "Valor Bruto (R$)":     parsed[1],
                         "Valor IR/IOF (R$)":    parsed[2],
                         "Valor Líquido (R$)":   parsed[3]})
    return pd.DataFrame(recs)

# ── Main ──────────────────────────────────────────────────────────────────

# %%

def main(caminho_pdf):
    pdf_path = Path(caminho_pdf)
    if not pdf_path.exists():
        print(f"Arquivo não encontrado: {caminho_pdf}"); sys.exit(1)

    print(f"Lendo: {pdf_path.name} ...")
    with pdfplumber.open(pdf_path) as pdf:
        texto = "\n".join(p.extract_text() or "" for p in pdf.pages)

    linhas = limpa_linhas(texto)
    df_res = extrair_resumo(linhas)
    df_mov = extrair_movimentacao(linhas)
    df_cdb = extrair_cdb(linhas)

    saida = pdf_path.stem + "_extraido.xlsx"
    with pd.ExcelWriter(saida, engine="openpyxl") as w:
        df_res.to_excel(w, sheet_name="Resumo",         index=False)
        df_mov.to_excel(w, sheet_name="Conta Corrente", index=False)
        df_cdb.to_excel(w, sheet_name="CDB-RDB",        index=False)

    print(f"\n✅ Arquivo salvo: {saida}")
    print(f"   Resumo:         {len(df_res)} linhas")
    print(f"   Conta Corrente: {len(df_mov)} transações")
    print(f"   CDB/RDB:        {len(df_cdb)} movimentações")
    return saida, df_res, df_mov, df_cdb

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python extrator_santander.py <caminho_do_pdf>"); sys.exit(1)
    main(sys.argv[1])


# Para rodar o código, digite no terminal: python Extrator.py Extrato_consolidado_mensal.pdf    

"""
Orquestrador: roda otimização Markowitz para a carteira BRL.

Pipeline:
1. Carrega definição da carteira (carteira_brl.py)
2. Baixa preços (data_loader: ANBIMA + BCB + yfinance)
3. Calcula CDI anualizado como taxa livre de risco
4. Roda Markowitz Monte Carlo
5. Imprime resumo dos portfólios ótimos
6. Salva gráfico em outputs/

Uso:
    cd Projetos/simulador_carteiras
    python -m scripts.run_otimizacao_brl
"""

import argparse
from datetime import date

from carteiras.carteira_brl import load_carteira, get_tickers
from core.data_loader import carregar_precos
from analysis.markowitz import calcular_fronteira_eficiente
from visualization.fronteira_eficiente import plotar_fronteira
from analysis.metricas import retorno_equivalente_cdi


# Caminho padrão quando o usuário não passa --config
CARTEIRA_DEFAULT = "carteiras/configs/balanceada.json"


def main() -> None:
    """Roda o pipeline completo de otimização."""
    
    # ========================================================
    # 0. PARSEAR ARGUMENTOS DA LINHA DE COMANDO
    # ========================================================
    parser = argparse.ArgumentParser(
        description="Roda otimização Markowitz para uma carteira BRL.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CARTEIRA_DEFAULT,
        help=f"Caminho do JSON de configuração da carteira "
             f"(default: {CARTEIRA_DEFAULT})",
    )
    args = parser.parse_args()
    
    # ========================================================
    # 1. CARREGAR CARTEIRA DO JSON
    # ========================================================
    carteira = load_carteira(args.config)
    
    print("\n" + "█" * 60)
    print(f"█ {carteira['nome'].upper()}")
    print("█" * 60)
    print(f"📂 Config: {args.config}")
    
    ativos = get_tickers(carteira)
    benchmark = carteira['benchmark']
    
    # TODO: tornar a janela configurável (argumentos CLI)
    data_inicio = carteira['data_inicio_default']
    data_fim = date.today()
    
    print(f"\n📋 Carteira: {len(ativos)} ativos + benchmark ({benchmark})")
    print(f"📅 Janela solicitada: {data_inicio} a {data_fim}")
    
    # ========================================================
    # 2. BAIXAR PREÇOS (ativos + benchmark)
    # ========================================================
    precos = carregar_precos(
        tickers=ativos + [benchmark],
        data_inicio=data_inicio,
        data_fim=data_fim,
    )
    
    # ========================================================
    # 3. CALCULAR TAXA LIVRE DE RISCO (CDI anualizado NA JANELA DO MARKOWITZ)
    # ========================================================
    # Identifica primeiro dia onde TODOS os ativos têm dados
    # (essa é a mesma janela que o Markowitz vai usar internamente)
    janela_inicio = precos[ativos].dropna().index.min()

    # Recorta CDI nessa janela pra calcular taxa anualizada COMPARÁVEL
    serie_cdi_alinhada = precos[benchmark].loc[janela_inicio:].dropna()
    n_anos = len(serie_cdi_alinhada) / 252
    cdi_anual = (
        serie_cdi_alinhada.iloc[-1] / serie_cdi_alinhada.iloc[0]
    ) ** (1/n_anos) - 1

    print(f"\n💰 CDI anualizado na janela ({janela_inicio.date()} → hoje): {cdi_anual:.2%}")
    
    # ========================================================
    # 4. RODAR MARKOWITZ
    # ========================================================
    resultado = calcular_fronteira_eficiente(
        precos=precos[ativos],
        taxa_livre_anual=cdi_anual,
        seed=42,
    )
    
    # ========================================================
    # 5. IMPRIMIR PESOS DOS PORTFÓLIOS ÓTIMOS
    # ========================================================
    _imprimir_pesos("MAX SHARPE", resultado['max_sharpe'], cdi_anual)
    _imprimir_pesos("MIN VARIÂNCIA", resultado['min_variancia'], cdi_anual)
    
    # ========================================================
    # 6. PLOTAR E SALVAR GRÁFICO
    # ========================================================
    print(f"\n📊 Gerando gráfico...")
    
    nome_arquivo = f"fronteira_eficiente_brl_{date.today().isoformat()}.png"
    
    plotar_fronteira(
        resultado=resultado,
        salvar=True,
        nome_arquivo=nome_arquivo,
        mostrar=True,
    )
    
    print(f"\n✅ Pipeline concluído com sucesso.\n")


def _imprimir_pesos(titulo: str, portfolio: dict, cdi_anual: float) -> None:
    """Imprime os pesos de um portfolio formatados."""
    print(f"\n" + "═" * 60)
    print(f"🎯 PORTFOLIO {titulo}")
    print("═" * 60)
    print(f"   Retorno anualizado:  {portfolio['retorno']:.2%}")
    print(f"   Volatilidade:        {portfolio['volatilidade']:.2%}")
    print(f"   Sharpe:              {portfolio['sharpe']:.2f}")
    
    # NOVO: retorno equivalente em formato CDI+X%
    eq = retorno_equivalente_cdi(portfolio['retorno'], cdi_anual)
    print(f"   Retorno Eq.:         {eq['formato_texto']}")
    
    print(f"\n   Composição:")
    
    # Ordena pesos do maior pro menor pra leitura mais fácil
    pesos_ordenados = sorted(
        portfolio['pesos'].items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    for ticker, peso in pesos_ordenados:
        # Visual: barra horizontal proporcional ao peso
        barra = '█' * int(peso * 40)  # max 40 chars
        print(f"   {ticker:<12} {peso:>6.2%}  {barra}")


if __name__ == "__main__":
    main()
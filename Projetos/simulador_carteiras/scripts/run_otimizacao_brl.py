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

from datetime import date

from carteiras.carteira_brl import CARTEIRA, get_tickers
from core.data_loader import carregar_precos
from analysis.markowitz import calcular_fronteira_eficiente
from visualization.fronteira_eficiente import plotar_fronteira


def main() -> None:
    """Roda o pipeline completo de otimização."""
    
    print("\n" + "█" * 60)
    print(f"█ {CARTEIRA['nome'].upper()}")
    print("█" * 60)
    
    # ========================================================
    # 1. CARREGAR DEFINIÇÕES
    # ========================================================
    ativos = get_tickers()
    benchmark = CARTEIRA['benchmark']
    
    # TODO: tornar a janela configurável (argumentos CLI)
    data_inicio = CARTEIRA['data_inicio_default']
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
    # 3. CALCULAR TAXA LIVRE DE RISCO (CDI anualizado)
    # ========================================================
    serie_cdi = precos[benchmark].dropna()
    n_anos = len(serie_cdi) / 252
    cdi_anual = (serie_cdi.iloc[-1] / serie_cdi.iloc[0]) ** (1/n_anos) - 1
    
    print(f"\n💰 CDI anualizado no período: {cdi_anual:.2%}")
    
    # ========================================================
    # 4. RODAR MARKOWITZ
    # ========================================================
    resultado = calcular_fronteira_eficiente(
        precos=precos[ativos],
        taxa_livre_anual=cdi_anual,
    )
    
    # ========================================================
    # 5. IMPRIMIR PESOS DOS PORTFÓLIOS ÓTIMOS
    # ========================================================
    _imprimir_pesos("MAX SHARPE", resultado['max_sharpe'])
    _imprimir_pesos("MIN VARIÂNCIA", resultado['min_variancia'])
    
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


def _imprimir_pesos(titulo: str, portfolio: dict) -> None:
    """Imprime os pesos de um portfolio formatados."""
    print(f"\n" + "═" * 60)
    print(f"🎯 PORTFOLIO {titulo}")
    print("═" * 60)
    print(f"   Retorno anualizado:  {portfolio['retorno']:.2%}")
    print(f"   Volatilidade:        {portfolio['volatilidade']:.2%}")
    print(f"   Sharpe:              {portfolio['sharpe']:.2f}")
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
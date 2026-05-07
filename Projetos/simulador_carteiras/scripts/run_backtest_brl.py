"""
Orquestrador: roda backtest histórico da carteira BRL.

Pipeline:
1. Carrega definição da carteira
2. Baixa preços históricos
3. Roda Markowitz pra obter pesos do Max Sharpe e Min Variância
4. Para cada portfolio + CDI, roda backtest com 4 estratégias
5. Imprime tabela comparativa
6. Plota evolução + drawdown das estratégias principais

Uso:
    cd Projetos/simulador_carteiras
    python -m scripts.run_backtest_brl
"""


from datetime import date
import pandas as pd
from carteiras.carteira_brl import CARTEIRA, get_tickers
from core.data_loader import carregar_precos
from analysis.markowitz import calcular_fronteira_eficiente
from analysis.backtest import executar_backtest
from visualization.backtest import plotar_backtest


VALOR_INICIAL = 100.0

# Estratégias a testar pra cada carteira
ESTRATEGIAS = ['buy_and_hold', 'mensal', 'trimestral', 'anual']


def main() -> None:
    """Pipeline completo de backtest histórico."""
    
    print("\n" + "█" * 60)
    print(f"█ BACKTEST HISTÓRICO — {CARTEIRA['nome'].upper()}")
    print("█" * 60)
    
    # ========================================================
    # 1. CARREGAR DADOS E RODAR MARKOWITZ
    # ========================================================
    ativos = get_tickers()
    benchmark = CARTEIRA['benchmark']
    
    precos = carregar_precos(
        tickers=ativos + [benchmark],
        data_inicio=CARTEIRA['data_inicio_default'],
        data_fim=date.today(),
    )
    
    # CDI anualizado pra Markowitz
    serie_cdi = precos[benchmark].dropna()
    n_anos = len(serie_cdi) / 252
    cdi_anual = (serie_cdi.iloc[-1] / serie_cdi.iloc[0]) ** (1/n_anos) - 1
    
    print(f"\n💰 CDI anualizado: {cdi_anual:.2%}")
    
    # Roda Markowitz pra extrair pesos ótimos
    resultado_markowitz = calcular_fronteira_eficiente(
        precos=precos[ativos],
        taxa_livre_anual=cdi_anual,
    )
    
    # ========================================================
    # 2. BACKTEST: MAX SHARPE COM 4 ESTRATÉGIAS
    # ========================================================
    print("\n" + "═" * 60)
    print("📊 BACKTEST: MAX SHARPE")
    print("═" * 60)
    
    pesos_max_sharpe = resultado_markowitz['max_sharpe']['pesos']
    backtests_max_sharpe = {}
    
    for estrategia in ESTRATEGIAS:
        print(f"\n   Rodando estratégia: {estrategia}...")
        backtests_max_sharpe[estrategia] = executar_backtest(
            precos=precos[ativos],
            pesos=pesos_max_sharpe,
            estrategia=estrategia,
            valor_inicial=VALOR_INICIAL,
        )
    
    # ========================================================
    # 3. BACKTEST: MIN VARIÂNCIA COM 4 ESTRATÉGIAS
    # ========================================================
    print("\n" + "═" * 60)
    print("📊 BACKTEST: MIN VARIÂNCIA")
    print("═" * 60)
    
    pesos_min_var = resultado_markowitz['min_variancia']['pesos']
    backtests_min_var = {}
    
    for estrategia in ESTRATEGIAS:
        print(f"\n   Rodando estratégia: {estrategia}...")
        backtests_min_var[estrategia] = executar_backtest(
            precos=precos[ativos],
            pesos=pesos_min_var,
            estrategia=estrategia,
            valor_inicial=VALOR_INICIAL,
        )
    
    # ========================================================
    # 4. BACKTEST: CDI (BENCHMARK)
    # ========================================================
    print("\n" + "═" * 60)
    print("📊 BACKTEST: CDI (benchmark)")
    print("═" * 60)
    
    backtest_cdi = executar_backtest(
        precos=precos[[benchmark]],
        pesos={benchmark: 1.0},
        estrategia='buy_and_hold',  # CDI não rebalanceia
        valor_inicial=VALOR_INICIAL,
    )
    
    # ========================================================
    # 5. TABELA COMPARATIVA
    # ========================================================
    _imprimir_tabela_comparativa(
        backtests_max_sharpe=backtests_max_sharpe,
        backtests_min_var=backtests_min_var,
        backtest_cdi=backtest_cdi,
    )
    
    # ========================================================
    # 6. GRÁFICO COMPARATIVO (estratégias principais)
    # ========================================================
    print(f"\n📊 Gerando gráfico comparativo...")
    
    # Pra o gráfico, escolhemos buy_and_hold de cada (mais "limpo" visualmente)
    resultados_grafico = {
        'Max Sharpe':    backtests_max_sharpe['buy_and_hold'],
        'Min Variância': backtests_min_var['buy_and_hold'],
        'CDI':           backtest_cdi,
    }
    
    # Alinha janelas pra comparação justa
    resultados_grafico = _alinhar_janelas(resultados_grafico, VALOR_INICIAL)
    
    nome_arquivo = f"backtest_brl_{date.today().isoformat()}.png"
    
    plotar_backtest(
        resultados=resultados_grafico,
        titulo=(
            f"Backtest: Max Sharpe vs Min Variância vs CDI\n"
            f"Valor inicial: R${VALOR_INICIAL:.0f} | "
            f"Estratégia: Buy and Hold"
        ),
        salvar=True,
        nome_arquivo=nome_arquivo,
        mostrar=True,
    )
    
    print(f"\n✅ Pipeline de backtest concluído.\n")


def _imprimir_tabela_comparativa(
    backtests_max_sharpe: dict,
    backtests_min_var: dict,
    backtest_cdi: dict,
) -> None:
    """Imprime tabela comparativa de todas as estratégias."""
    
    print("\n" + "═" * 90)
    print("📋 TABELA COMPARATIVA: TODAS AS ESTRATÉGIAS")
    print("═" * 90)
    
    cabecalho = (
        f"{'Carteira':<15} {'Estratégia':<14} "
        f"{'Ret.Anual':>10} {'Volat':>8} {'MaxDD':>8} "
        f"{'Valor Final':>12} {'Custos':>10}"
    )
    print(cabecalho)
    print("─" * 90)
    
    # Max Sharpe
    for estrategia, resultado in backtests_max_sharpe.items():
        _imprimir_linha("Max Sharpe", estrategia, resultado)
    
    print()
    
    # Min Variância
    for estrategia, resultado in backtests_min_var.items():
        _imprimir_linha("Min Variância", estrategia, resultado)
    
    print()
    
    # CDI
    _imprimir_linha("CDI", "buy_and_hold", backtest_cdi)
    
    print("═" * 90)
    print("\n💡 Observações:")
    print("   • Custos = corretagem + IR pagos em rebalanceamentos")
    print("   • Buy and Hold tem custos = 0 (sem rebalanceamento)")
    print("   • Modelo simplificado: aliquota IR única, sem isenções")


def _imprimir_linha(carteira: str, estrategia: str, resultado: dict) -> None:
    """Imprime uma linha da tabela comparativa."""
    valor_final = resultado['valores'].iloc[-1]
    custos = resultado.get('custos', {}).get('total_custos', 0.0)
    
    print(
        f"{carteira:<15} {estrategia:<14} "
        f"{resultado['retorno_anualizado']:>9.2%} "
        f"{resultado['volatilidade_anualizada']:>7.2%} "
        f"{resultado['drawdown_maximo']:>7.2%} "
        f"R${valor_final:>9.2f} "
        f"R${custos:>8.2f}"
    )

def _alinhar_janelas(resultados: dict[str, dict], valor_inicial: float) -> dict[str, dict]:
    """
    Alinha múltiplos resultados de backtest pra começarem na mesma data.
    
    Encontra a data de início COMUM (a mais tardia entre as séries) e:
    - Recorta cada série a partir dessa data
    - Reescala todas pra começarem no mesmo valor inicial
    - Recalcula drawdown a partir do novo início (pico se reseta)
    
    Args:
        resultados: Dict {nome: resultado_backtest}
        valor_inicial: Valor inicial pra reescalar todas as séries
    
    Returns:
        Dict com mesma estrutura, mas séries alinhadas e reescaladas.
    """
    # Encontra a data de início mais tardia (= primeiro ponto em comum)
    data_inicio_comum = max(
        resultado['valores'].index.min()
        for resultado in resultados.values()
    )
    
    print(f"\n🔧 Alinhando janelas: todas as séries começam em {data_inicio_comum.date()}")
    
    resultados_alinhados = {}
    
    for nome, resultado in resultados.items():
        # 1. Recorta a partir da data comum
        valores_recortados = resultado['valores'].loc[data_inicio_comum:]
        
        # 2. Reescala pra começar no valor_inicial
        valor_no_inicio = valores_recortados.iloc[0]
        fator_reescala = valor_inicial / valor_no_inicio
        valores_alinhados = valores_recortados * fator_reescala
        
        # 3. Recalcula drawdown (pico se "reseta" no novo início)
        pico = valores_alinhados.cummax()
        drawdown_alinhado = (valores_alinhados - pico) / pico
        
        # Mantém o resto do resultado original, mas com valores recortados
        resultados_alinhados[nome] = {
            **resultado,  # copia tudo
            'valores': valores_alinhados,
            'drawdown': drawdown_alinhado,
        }
    
    return resultados_alinhados


if __name__ == "__main__":
    main()
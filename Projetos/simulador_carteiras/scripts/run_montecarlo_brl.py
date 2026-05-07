"""
Orquestrador: roda Monte Carlo de cenários futuros pra carteira BRL.

Pipeline:
1. Carrega definição da carteira
2. Baixa preços históricos
3. Roda Markowitz pra obter pesos do Max Sharpe e Min Variância
4. Para cada portfolio:
   a. Roda Bootstrap (10k simulações × 252 dias)
   b. Roda Normal multivariada (10k × 252 dias)
   c. Aplica pesos e calcula trajetórias
5. Plota grade comparativa 2x2

Uso:
    cd Projetos/simulador_carteiras
    python -m scripts.run_montecarlo_brl
"""

from datetime import date

import numpy as np

from carteiras.carteira_brl import CARTEIRA, get_tickers
from core.data_loader import carregar_precos
from analysis.markowitz import calcular_fronteira_eficiente
from analysis.monte_carlo_bootstrap import simular_bootstrap
from analysis.monte_carlo_normal import simular_normal
from analysis._montecarlo_utils import aplicar_pesos_e_acumular
from visualization.monte_carlo import plotar_comparativo


VALOR_INICIAL = 100.0  # carteira começa em 100 (visual mais limpo que 1.0)


def main() -> None:
    """Pipeline completo de Monte Carlo."""
    
    print("\n" + "█" * 60)
    print(f"█ MONTE CARLO — {CARTEIRA['nome'].upper()}")
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
    
    # CDI anualizado
    serie_cdi = precos[benchmark].dropna()
    n_anos = len(serie_cdi) / 252
    cdi_anual = (serie_cdi.iloc[-1] / serie_cdi.iloc[0]) ** (1/n_anos) - 1
    
    print(f"\n💰 CDI anualizado: {cdi_anual:.2%}")
    
    # Roda Markowitz
    resultado_markowitz = calcular_fronteira_eficiente(
        precos=precos[ativos],
        taxa_livre_anual=cdi_anual,
    )
    
    # ========================================================
    # 2. PREPARAR RETORNOS HISTÓRICOS PRA SIMULAÇÃO
    # ========================================================
    # Filtra janela com todos os dados e calcula retornos diários
    retornos_historicos = precos[ativos].dropna().pct_change().dropna()
    
    print(f"\n📊 Histórico pra simulação: {len(retornos_historicos)} dias úteis")
    
    # ========================================================
    # 3. RODA AMBOS OS MOTORES
    # ========================================================
    print("\n" + "═" * 60)
    print("🎲 RODANDO SIMULAÇÕES DE MONTE CARLO")
    print("═" * 60)
    
    # Mesmas seeds pros 2 métodos = comparação justa
    SEED_BOOTSTRAP = 42
    SEED_NORMAL = 42
    
    # Bootstrap
    print()
    retornos_boot = simular_bootstrap(
        retornos_historicos=retornos_historicos,
        seed=SEED_BOOTSTRAP,
    )
    
    # Normal
    print()
    retornos_normal = simular_normal(
        retornos_historicos=retornos_historicos,
        seed=SEED_NORMAL,
    )
    
    # ========================================================
    # 4. APLICAR PESOS DE CADA CARTEIRA
    # ========================================================
    # Pega pesos como vetor numpy (na ordem dos ativos)
    pesos_max_sharpe = np.array([
        resultado_markowitz['max_sharpe']['pesos'][a] for a in ativos
    ])
    pesos_min_var = np.array([
        resultado_markowitz['min_variancia']['pesos'][a] for a in ativos
    ])
    
    # Calcula trajetórias acumuladas pra cada combinação
    trajetorias = {
        'Max Sharpe': {
            'Bootstrap': aplicar_pesos_e_acumular(retornos_boot, pesos_max_sharpe, VALOR_INICIAL),
            'Normal':    aplicar_pesos_e_acumular(retornos_normal, pesos_max_sharpe, VALOR_INICIAL),
        },
        'Min Variância': {
            'Bootstrap': aplicar_pesos_e_acumular(retornos_boot, pesos_min_var, VALOR_INICIAL),
            'Normal':    aplicar_pesos_e_acumular(retornos_normal, pesos_min_var, VALOR_INICIAL),
        },
    }
    
    # ========================================================
    # 5. RESUMO ESTATÍSTICO
    # ========================================================
    print("\n" + "═" * 60)
    print(f"📈 RESUMO DAS PROJEÇÕES (após 1 ano)")
    print("═" * 60)
    
    for nome_carteira, metodos in trajetorias.items():
        print(f"\n🎯 {nome_carteira}")
        for nome_metodo, traj in metodos.items():
            valores_finais = traj[:, -1]
            p5, p50, p95 = np.percentile(valores_finais, [5, 50, 95])
            print(
                f"   {nome_metodo:<10} "
                f"P5: {p5:.1f} ({(p5/VALOR_INICIAL - 1):+.1%}) | "
                f"P50: {p50:.1f} ({(p50/VALOR_INICIAL - 1):+.1%}) | "
                f"P95: {p95:.1f} ({(p95/VALOR_INICIAL - 1):+.1%})"
            )
    
    # ========================================================
    # 6. GERAR GRÁFICO COMPARATIVO
    # ========================================================
    print(f"\n📊 Gerando gráfico comparativo...")
    
    nome_arquivo = f"monte_carlo_brl_{date.today().isoformat()}.png"
    
    plotar_comparativo(
        trajetorias_dict=trajetorias,
        valor_inicial=VALOR_INICIAL,
        salvar=True,
        nome_arquivo=nome_arquivo,
        mostrar=True,
    )
    
    print(f"\n✅ Pipeline Monte Carlo concluído.\n")


if __name__ == "__main__":
    main()
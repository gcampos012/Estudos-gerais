"""
Visualização comparativa de Monte Carlo (Bootstrap vs Normal).

Plota trajetórias projetadas com bandas de percentis para diferentes
carteiras e métodos, em uma grade de subplots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.config import OUTPUTS_DIR


# Percentis a destacar nas bandas
PERCENTIS = [5, 25, 50, 75, 95]


def _plotar_carteira_unica(
    ax: plt.Axes,
    trajetorias: np.ndarray,
    titulo: str,
    cor_base: str,
    valor_inicial: float = 1.0,
) -> None:
    """
    Plota UMA combinação carteira+método em um subplot.
    
    Args:
        ax: Eixo do matplotlib (subplot)
        trajetorias: Array (n_simulacoes, n_dias)
        titulo: Título do subplot
        cor_base: Cor principal (hex ou nome)
        valor_inicial: Valor inicial da carteira (pra marcar referência)
    """
    n_simulacoes, n_dias = trajetorias.shape
    
    # Eixo X: dias futuros (1 a n_dias)
    dias = np.arange(1, n_dias + 1)
    
    # Calcula percentis ao longo do tempo
    # axis=0 = "ao longo das simulações" → resulta num vetor de tamanho n_dias
    percentis_calc = np.percentile(trajetorias, PERCENTIS, axis=0)
    # percentis_calc shape: (5, n_dias) — uma linha por percentil
    
    # ============================================================
    # AMOSTRA DE TRAJETÓRIAS INDIVIDUAIS (textura visual)
    # ============================================================
    n_amostras = min(50, n_simulacoes)
    indices_amostra = np.random.choice(n_simulacoes, n_amostras, replace=False)
    
    for idx in indices_amostra:
        ax.plot(
            dias,
            trajetorias[idx],
            color=cor_base,
            alpha=0.05,           # quase transparente
            linewidth=0.5,
        )
    
    # ============================================================
    # BANDAS DE PERCENTIS
    # ============================================================
    # Banda P5-P95 (mais ampla, mais clara)
    ax.fill_between(
        dias,
        percentis_calc[0],   # P5
        percentis_calc[4],   # P95
        color=cor_base,
        alpha=0.2,
        label='P5 - P95 (90% dos cenários)',
    )
    
    # Banda P25-P75 (mais estreita, mais escura)
    ax.fill_between(
        dias,
        percentis_calc[1],   # P25
        percentis_calc[3],   # P75
        color=cor_base,
        alpha=0.35,
        label='P25 - P75 (50% dos cenários)',
    )
    
    # Linha da mediana
    ax.plot(
        dias,
        percentis_calc[2],   # P50
        color=cor_base,
        linewidth=2,
        label='Mediana (P50)',
    )
    
    # Linha de referência (valor inicial)
    ax.axhline(
        y=valor_inicial,
        color='gray',
        linestyle=':',
        linewidth=1,
        alpha=0.7,
        label=f'Valor inicial ({valor_inicial})',
    )
    
    # Anotações com valores finais (no último dia)
    valor_p5_final = percentis_calc[0, -1]
    valor_p50_final = percentis_calc[2, -1]
    valor_p95_final = percentis_calc[4, -1]
    
    texto = (
        f"Após {n_dias} dias:\n"
        f"  P5:  {valor_p5_final:.2f} ({(valor_p5_final/valor_inicial - 1):+.1%})\n"
        f"  P50: {valor_p50_final:.2f} ({(valor_p50_final/valor_inicial - 1):+.1%})\n"
        f"  P95: {valor_p95_final:.2f} ({(valor_p95_final/valor_inicial - 1):+.1%})"
    )
    
    ax.text(
        0.02, 0.98,
        texto,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor='white',
            edgecolor=cor_base,
            linewidth=0.8,
            alpha=0.9,
        ),
    )
    
    # Formatação
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_xlabel('Dias úteis no futuro', fontsize=9)
    ax.set_ylabel('Valor do portfolio', fontsize=9)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)


def plotar_comparativo(
    trajetorias_dict: dict[str, dict[str, np.ndarray]],
    valor_inicial: float = 1.0,
    titulo_geral: str | None = None,
    salvar: bool = True,
    nome_arquivo: str = "monte_carlo_comparativo.png",
    mostrar: bool = True,
) -> Path | None:
    """
    Plota grade comparativa de Monte Carlo: carteiras × métodos.
    
    Args:
        trajetorias_dict: Dict aninhado:
            {
                'Max Sharpe': {
                    'Bootstrap': array(...),
                    'Normal':    array(...),
                },
                'Min Variância': {
                    'Bootstrap': array(...),
                    'Normal':    array(...),
                },
            }
        valor_inicial: Valor inicial das carteiras (pra referência)
        titulo_geral: Título do gráfico inteiro
        salvar: Se True, salva PNG
        nome_arquivo: Nome do arquivo
        mostrar: Se True, abre janela
    
    Returns:
        Path do arquivo salvo (ou None)
    """
    carteiras = list(trajetorias_dict.keys())
    metodos = list(next(iter(trajetorias_dict.values())).keys())
    
    n_carteiras = len(carteiras)
    n_metodos = len(metodos)
    
    # Cria grade de subplots
    fig, axes = plt.subplots(
        n_carteiras,
        n_metodos,
        figsize=(7 * n_metodos, 5 * n_carteiras),
        sharey='row',   # subplots da mesma linha compartilham eixo Y
    )
    
    # Garante que axes seja sempre 2D (mesmo com 1 carteira ou 1 método)
    if n_carteiras == 1:
        axes = np.array([axes])
    if n_metodos == 1:
        axes = axes.reshape(-1, 1)
    
    # Cores por método (consistência visual)
    cores_metodo = {
        'Bootstrap': '#2874A6',   # azul institucional
        'Normal':    '#C0392B',   # vermelho terroso
    }
    
    # Plota cada combinação
    for i, carteira in enumerate(carteiras):
        for j, metodo in enumerate(metodos):
            trajetorias = trajetorias_dict[carteira][metodo]
            cor = cores_metodo.get(metodo, 'gray')
            
            _plotar_carteira_unica(
                ax=axes[i, j],
                trajetorias=trajetorias,
                titulo=f"{carteira} — {metodo}",
                cor_base=cor,
                valor_inicial=valor_inicial,
            )
    
    # Título geral
    if titulo_geral is None:
        n_dias = trajetorias.shape[1]
        n_simulacoes = trajetorias.shape[0]
        titulo_geral = (
            f"Monte Carlo: Projeção de {n_dias} dias úteis "
            f"({n_simulacoes:,} simulações por cenário)"
        )
    
    fig.suptitle(titulo_geral, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Salvar / mostrar
    caminho_salvo = None
    if salvar:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        caminho_salvo = OUTPUTS_DIR / nome_arquivo
        plt.savefig(caminho_salvo, dpi=150, bbox_inches='tight')
        print(f"💾 Gráfico salvo em: {caminho_salvo}")
    
    if mostrar:
        plt.show()
    else:
        plt.close()
    
    return caminho_salvo
"""
Visualização da fronteira eficiente de Markowitz.

Gera gráfico clássico com scatter de portfólios coloridos por Sharpe,
marcadores nos pontos ótimos (max Sharpe, min variância) e linha CML.

Recebe o resultado de analysis.markowitz.calcular_fronteira_eficiente().
"""

# ============================================================
# BLOCO 1 - IMPORTS 
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import OUTPUTS_DIR

# ============================================================
# BLOCO 2 - PLOTAGEM DA FRONTEIRA EFICIENTE
# ============================================================

def plotar_fronteira(
    resultado: dict,
    titulo: str | None = None,
    salvar: bool = True,
    nome_arquivo: str = "fronteira_eficiente.png",
    mostrar: bool = True,
) -> Path | None:
    """
    Plota a fronteira eficiente de Markowitz.
    
    Args:
        resultado: Dict retornado por calcular_fronteira_eficiente()
        titulo: Título customizado do gráfico (default: gerado automaticamente)
        salvar: Se True, salva PNG em outputs/
        nome_arquivo: Nome do arquivo PNG (se salvar=True)
        mostrar: Se True, abre janela com o gráfico
    
    Returns:
        Path do arquivo salvo (se salvar=True), senão None.
    """
    df_portfolios = resultado['portfolios']
    max_sharpe = resultado['max_sharpe']
    min_variancia = resultado['min_variancia']
    janela = resultado['janela']
    
    # ============================================================
    # SETUP DA FIGURA
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # ============================================================
    # SCATTER DE PORTFÓLIOS (colorido por Sharpe)
    # ============================================================
    scatter = ax.scatter(
        df_portfolios['volatilidade'],
        df_portfolios['retorno'],
        c=df_portfolios['sharpe'],
        cmap='viridis',           # paleta de cores (verde→amarelo)
        s=10,                     # tamanho dos pontos
        alpha=0.5,                # transparência
        edgecolor='none',
    )
    
    # Barra de cores
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=11)
    
    # ============================================================
    # MARCADOR: MAX SHARPE (diamante elegante)
    # ============================================================
    ax.scatter(
        max_sharpe['volatilidade'],
        max_sharpe['retorno'],
        marker='D',
        s=120,
        c='white',
        edgecolor='#C0392B',          # vermelho terroso, sóbrio
        linewidth=2,
        label=f"Max Sharpe ({max_sharpe['sharpe']:.2f})",
        zorder=5,
    )

    # Anotação Max Sharpe
    ax.annotate(
        f"  Max Sharpe\n  Ret: {max_sharpe['retorno']:.1%}\n  Vol: {max_sharpe['volatilidade']:.1%}",
        xy=(max_sharpe['volatilidade'], max_sharpe['retorno']),
        xytext=(15, 10),
        textcoords='offset points',
        fontsize=9,
        color='#C0392B',
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor='white',
            edgecolor='#C0392B',
            linewidth=0.8,
            alpha=0.9,
        ),
    )

    # ============================================================
    # MARCADOR: MIN VARIÂNCIA (círculo elegante)
    # ============================================================
    ax.scatter(
        min_variancia['volatilidade'],
        min_variancia['retorno'],
        marker='o',
        s=120,
        c='white',
        edgecolor='#2874A6',          # azul institucional
        linewidth=2,
        label=f"Min Variância (vol {min_variancia['volatilidade']:.2%})",
        zorder=5,
    )

    # Anotação Min Variância
    ax.annotate(
        f"  Min Variância\n  Ret: {min_variancia['retorno']:.1%}\n  Vol: {min_variancia['volatilidade']:.1%}",
        xy=(min_variancia['volatilidade'], min_variancia['retorno']),
        xytext=(15, -30),
        textcoords='offset points',
        fontsize=9,
        color='#2874A6',
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor='white',
            edgecolor='#2874A6',
            linewidth=0.8,
            alpha=0.9,
        ),
    ) 
    
    # ============================================================
    # LINHA CML (CAPITAL MARKET LINE)
    # ============================================================
    # CML vai do (0, taxa_livre) passando pelo Max Sharpe
    taxa_livre = max_sharpe['retorno'] - max_sharpe['sharpe'] * max_sharpe['volatilidade']
    
    x_cml = np.linspace(0, df_portfolios['volatilidade'].max() * 1.1, 100)
    y_cml = taxa_livre + max_sharpe['sharpe'] * x_cml
    
    ax.plot(
        x_cml,
        y_cml,
        '--',
        color='red',
        alpha=0.5,
        linewidth=1.5,
        label='Capital Market Line',
    )
    
    # ============================================================
    # FORMATAÇÃO DOS EIXOS
    # ============================================================
    ax.set_xlabel('Volatilidade Anualizada (Risco)', fontsize=12)
    ax.set_ylabel('Retorno Anualizado', fontsize=12)
    
    # Formato percentual
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # ============================================================
    # TÍTULO
    # ============================================================
    if titulo is None:
        titulo = (
            f"Fronteira Eficiente de Markowitz\n"
            f"Período: {janela['inicio']} a {janela['fim']} "
            f"({janela['n_dias']} dias úteis) | "
            f"{len(df_portfolios):,} simulações"
        )
    ax.set_title(titulo, fontsize=13)
    
    # Legenda
    ax.legend(loc='lower right', fontsize=10)
    
    # Ajusta layout
    plt.tight_layout()
    
    # ============================================================
    # SALVAR / MOSTRAR
    # ============================================================
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
"""
Visualização de backtest histórico.

Gera gráfico em 2 painéis:
- Superior: evolução do patrimônio das carteiras + benchmark
- Inferior: drawdown ao longo do tempo
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from core.config import OUTPUTS_DIR


# Paleta de cores consistente (mesmas usadas em outros gráficos)
CORES = {
    'Max Sharpe':    '#C0392B',  # vermelho terroso
    'Min Variância': '#2874A6',  # azul institucional
    'CDI':           '#7F8C8D',  # cinza médio (benchmark)
}

# Estilos por carteira
ESTILOS = {
    'Max Sharpe':    {'linestyle': '-',  'linewidth': 2.0, 'alpha': 1.0},
    'Min Variância': {'linestyle': '-',  'linewidth': 2.0, 'alpha': 1.0},
    'CDI':           {'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.8},
}


def plotar_backtest(
    resultados: dict[str, dict],
    titulo: str | None = None,
    salvar: bool = True,
    nome_arquivo: str = "backtest.png",
    mostrar: bool = True,
) -> Path | None:
    """
    Plota backtest comparativo em 2 painéis.
    
    Args:
        resultados: Dict {nome_carteira: dict_resultado_de_executar_backtest}
                    Cada valor deve conter ao menos 'valores' e 'drawdown'.
        titulo: Título do gráfico (default: gerado automaticamente)
        salvar: Se True, salva PNG em outputs/
        nome_arquivo: Nome do arquivo PNG
        mostrar: Se True, abre janela com gráfico
    
    Returns:
        Path do arquivo salvo (ou None)
    """
    # Cria figura com 2 subplots compartilhando eixo X
    fig, (ax_valor, ax_dd) = plt.subplots(
        2, 1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},  # painel valor 2x maior que drawdown
    )
    
    # ============================================================
    # PAINEL 1: EVOLUÇÃO DO PATRIMÔNIO
    # ============================================================
    for nome, resultado in resultados.items():
        cor = CORES.get(nome, 'black')
        estilo = ESTILOS.get(nome, {'linestyle': '-', 'linewidth': 1.5, 'alpha': 1.0})
        
        ax_valor.plot(
            resultado['valores'].index,
            resultado['valores'].values,
            label=nome,
            color=cor,
            **estilo,
        )
    
    # Linha horizontal do valor inicial (referência visual)
    primeiro_resultado = next(iter(resultados.values()))
    valor_inicial = primeiro_resultado['valores'].iloc[0]
    
    ax_valor.axhline(
        y=valor_inicial,
        color='gray',
        linestyle=':',
        linewidth=1,
        alpha=0.5,
    )
    
    ax_valor.set_ylabel('Valor do Portfolio (R$)', fontsize=11)
    ax_valor.set_title('Evolução do Patrimônio', fontsize=12, fontweight='bold')
    ax_valor.legend(loc='upper left', fontsize=10)
    ax_valor.grid(True, alpha=0.3)
    
    # ============================================================
    # PAINEL 2: DRAWDOWN
    # ============================================================
    for nome, resultado in resultados.items():
        cor = CORES.get(nome, 'black')
        estilo = ESTILOS.get(nome, {'linestyle': '-', 'linewidth': 1.5, 'alpha': 1.0})
        
        ax_dd.fill_between(
            resultado['drawdown'].index,
            resultado['drawdown'].values * 100,  # converte pra %
            0,
            color=cor,
            alpha=0.3,
        )
        ax_dd.plot(
            resultado['drawdown'].index,
            resultado['drawdown'].values * 100,
            color=cor,
            label=nome,
            **estilo,
        )
    
    # Linha do zero
    ax_dd.axhline(y=0, color='black', linewidth=0.5)
    
    ax_dd.set_ylabel('Drawdown (%)', fontsize=11)
    ax_dd.set_xlabel('Data', fontsize=11)
    ax_dd.set_title('Drawdown abaixo do pico histórico', fontsize=12, fontweight='bold')
    ax_dd.grid(True, alpha=0.3)
    
    # ============================================================
    # FORMATAÇÃO COMPARTILHADA
    # ============================================================
    # Formato do eixo X (datas)
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Rotação dos labels do eixo X
    plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=45)
    
    # ============================================================
    # TÍTULO GERAL
    # ============================================================
    if titulo is None:
        # Pega janela do primeiro resultado
        primeiro = next(iter(resultados.values()))
        data_inicio = primeiro['valores'].index.min().date()
        data_fim = primeiro['valores'].index.max().date()
        titulo = f"Backtest Histórico: {data_inicio} a {data_fim}"
    
    fig.suptitle(titulo, fontsize=14, fontweight='bold', y=1.00)
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
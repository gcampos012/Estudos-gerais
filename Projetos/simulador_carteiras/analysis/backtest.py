"""
Backtest histórico de carteiras.

Implementa 4 estratégias:
- Buy and Hold (estático)
- Rebalanceamento mensal
- Rebalanceamento trimestral
- Rebalanceamento anual

Calcula a evolução do patrimônio e métricas de performance.
"""

# ============================================================
# BLOCO 1 - IMPORTS E HELPER DE DRAWDOWN
# ============================================================

from typing import Literal

import numpy as np
import pandas as pd

from core.config import DIAS_UTEIS_ANO
from analysis.metricas import (
    retorno_anualizado,
    volatilidade_anualizada,
    sharpe_ratio,
)

# ============================================================
# CONSTANTES DE CUSTOS (modelo simplificado)
# ============================================================

# Taxa de corretagem aplicada a cada operação (compra ou venda)
# Inclui spread bid-ask em ETFs/ações
TAXA_CORRETAGEM_DEFAULT = 0.0030  # 0.30%

# Alíquota de IR sobre ganhos realizados em vendas
# Simplificação: aplicado uniformemente (sem isenção, sem tabela regressiva)
ALIQUOTA_IR_DEFAULT = 0.15  # 15%


# Tipo das estratégias permitidas (validação automática)
Estrategia = Literal['buy_and_hold', 'mensal', 'trimestral', 'anual']


def _calcular_drawdown_serie(valores: pd.Series) -> pd.Series:
    """
    Calcula o drawdown ao longo do tempo (não apenas o máximo).
    
    Args:
        valores: Série de valores do portfolio ao longo do tempo
    
    Returns:
        Série de mesma dimensão, valores em [-1, 0]
        Ex: -0.20 = portfolio está 20% abaixo do pico histórico
    """
    pico = valores.cummax()
    drawdown = (valores - pico) / pico
    return drawdown

# ============================================================
# BLOCO 2 - BUY AND HOLD (sem rebalanceamento)
# ============================================================

def _backtest_buy_and_hold(
    retornos: pd.DataFrame,
    pesos: dict[str, float],
    valor_inicial: float,
) -> pd.Series:
    """
    Backtest buy-and-hold (sem rebalanceamento).
    
    Cada ativo evolui independente; pesos vão se deslocando.
    
    Args:
        retornos: DataFrame de retornos diários (índice=data, colunas=ativos)
        pesos: Dict {ticker: peso_inicial}
        valor_inicial: Valor inicial do portfolio
    
    Returns:
        Series com valor total do portfolio ao longo do tempo.
    """
    # Garante mesma ordem de colunas
    pesos_array = np.array([pesos[ticker] for ticker in retornos.columns])
    
    # Valor inicial alocado em cada ativo
    valores_iniciais = valor_inicial * pesos_array
    
    # Evolução de cada ativo (vetorizado): cumprod dos (1+r)
    fatores_acumulados = (1 + retornos).cumprod()
    
    # valor_ativo[t,i] = valor_inicial[i] * fator_acumulado[t,i]
    # Multiplicação broadcast: (T,n) * (n,) → (T,n)
    valores_ativos = fatores_acumulados * valores_iniciais
    
    # Soma os ativos pra ter valor total do portfolio
    return valores_ativos.sum(axis=1)

# ============================================================
# BLOCO 3 - REBALANCEAMENTO PERIÓDICO
# ============================================================

# Mapeia frequência → string que pandas entende pra "regra de fim de período"
FREQUENCIAS_PANDAS = {
    'mensal':     'ME',   # Month End
    'trimestral': 'QE',   # Quarter End
    'anual':      'YE',   # Year End
}

def _backtest_rebalanceamento(
    retornos: pd.DataFrame,
    pesos: dict[str, float],
    valor_inicial: float,
    frequencia: Literal['mensal', 'trimestral', 'anual'],
    taxa_corretagem: float = TAXA_CORRETAGEM_DEFAULT,
    aliquota_ir: float = ALIQUOTA_IR_DEFAULT,
) -> tuple[pd.Series, dict]:
    """
    Backtest com rebalanceamento periódico (com custos e impostos).
    """
    pesos_array = np.array([pesos[ticker] for ticker in retornos.columns])
    freq_pandas = FREQUENCIAS_PANDAS[frequencia]
    ativos = list(retornos.columns)
    
    # ============================================================
    # IDENTIFICA FINS DE PERÍODO USANDO DIAS ÚTEIS REAIS
    # ============================================================
    # Em vez de pegar fim do calendário (que pode cair em fim de semana),
    # agrupamos por período e pegamos o ÚLTIMO DIA ÚTIL DISPONÍVEL de cada um.
    # Solução: criar uma série com a posição de cada dia, e pegar o max() por período.
    posicoes = pd.Series(range(len(retornos)), index=retornos.index)
    posicoes_fim_periodo = posicoes.resample(freq_pandas).max().dropna().astype(int)
    
    # pontos_rebal agora são índices REAIS do DataFrame
    pontos_rebal = retornos.index[posicoes_fim_periodo.values]
    
    valor_ativo = valor_inicial * pesos_array
    preco_medio = np.ones(len(ativos))
    
    valores_carteira = pd.Series(index=retornos.index, dtype=float)
    
    total_corretagem = 0.0
    total_ir = 0.0
    n_rebalanceamentos = 0
    
    indice_inicio_periodo = retornos.index[0]
    
    for i_periodo, fim_periodo in enumerate(pontos_rebal):
        retornos_periodo = retornos.loc[indice_inicio_periodo:fim_periodo]
        
        if len(retornos_periodo) == 0:
            continue
        
        # 1. ACUMULA RETORNOS NO PERÍODO
        fatores = (1 + retornos_periodo).cumprod().values
        valores_no_periodo = fatores * valor_ativo
        valores_totais_periodo = valores_no_periodo.sum(axis=1)
        
        valores_carteira.loc[retornos_periodo.index] = valores_totais_periodo
        
        # 2. ATUALIZA ESTADO PRO FIM DO PERÍODO
        valor_ativo = valores_no_periodo[-1]
        valor_total = valor_ativo.sum()
        
        # 3. SE NÃO É O ÚLTIMO PERÍODO, REBALANCEIA
        eh_ultimo_periodo = (i_periodo == len(pontos_rebal) - 1)
        
        if not eh_ultimo_periodo:
            valor_desejado = valor_total * pesos_array
            delta = valor_desejado - valor_ativo
            
            volume_movimentado = np.abs(delta).sum()
            corretagem = volume_movimentado * taxa_corretagem
            
            ir_periodo = 0.0
            for j in range(len(ativos)):
                if delta[j] < 0:
                    valor_vendido = -delta[j]
                    fator_atual = valor_ativo[j] / preco_medio[j] if preco_medio[j] > 0 else 1.0
                    if fator_atual > 1.0:
                        ganho_proporcional = 1 - (1 / fator_atual)
                        ganho = valor_vendido * ganho_proporcional
                        ir_periodo += ganho * aliquota_ir
            
            custo_total = corretagem + ir_periodo
            valor_total_liquido = valor_total - custo_total
            
            valor_ativo = valor_total_liquido * pesos_array
            preco_medio = valor_ativo.copy()
            
            # Atualiza o valor da Series no fim do período (subtrai custos)
            # IMPORTANTE: usar .loc com índice válido (já garantido)
            valores_carteira.loc[fim_periodo] = valor_total_liquido
            
            total_corretagem += corretagem
            total_ir += ir_periodo
            n_rebalanceamentos += 1
            
            # Próximo período: dia seguinte ao fim atual
            idx_pos = retornos.index.get_loc(fim_periodo) + 1
            if idx_pos < len(retornos.index):
                indice_inicio_periodo = retornos.index[idx_pos]
    
    custos_info = {
        'total_corretagem': total_corretagem,
        'total_ir': total_ir,
        'total_custos': total_corretagem + total_ir,
        'n_rebalanceamentos': n_rebalanceamentos,
    }
    
    return valores_carteira.dropna(), custos_info

# ============================================================
# BLOCO 4 - FUNÇÃO PÚBLICA - DESPACHANTE DE ESTRATÉGIAS
# ============================================================

def executar_backtest(
    precos: pd.DataFrame,
    pesos: dict[str, float],
    estrategia: Estrategia,
    valor_inicial: float = 100.0,
    taxa_corretagem: float = TAXA_CORRETAGEM_DEFAULT,
    aliquota_ir: float = ALIQUOTA_IR_DEFAULT,
) -> dict:
    """
    Executa backtest histórico de uma carteira.
    
    ⚠️ MODELO SIMPLIFICADO DE CUSTOS:
    - Custo: aplicado sobre volume movimentado em rebalanceamentos
    - IR: alíquota fixa sobre ganhos, sem isenções, sem tabela regressiva
    - Preço médio é resetado a cada rebalanceamento (subestima IR)
    
    Para análise relativa entre estratégias, esses números são informativos.
    Para decisão de investimento real, considerar modelagem fiscal mais detalhada.
    
    Args:
        precos: DataFrame de preços/níveis
        pesos: Dict {ticker: peso}
        estrategia: 'buy_and_hold', 'mensal', 'trimestral', 'anual'
        valor_inicial: Valor inicial (default: 100)
        taxa_corretagem: % por operação (default: 0.3%)
        aliquota_ir: % sobre ganhos (default: 15%)
    
    Returns:
        Dict com valores, drawdown, métricas e custos.
    """
    soma_pesos = sum(pesos.values())
    if not np.isclose(soma_pesos, 1.0, atol=0.01):
        raise ValueError(f"Pesos não somam 1.0 (soma={soma_pesos:.4f})")
    
    precos_validos = precos.dropna()
    ativos_disponiveis = set(precos_validos.columns)
    pesos_filtrados = {k: v for k, v in pesos.items() if k in ativos_disponiveis}
    
    if len(pesos_filtrados) < len(pesos):
        ausentes = set(pesos.keys()) - ativos_disponiveis
        print(f"⚠️  Ativos sem dados, ignorando: {ausentes}")
    
    retornos = precos_validos[list(pesos_filtrados.keys())].pct_change().dropna()
    
    # Despacha pra função certa
    if estrategia == 'buy_and_hold':
        valores = _backtest_buy_and_hold(retornos, pesos_filtrados, valor_inicial)
        custos_info = {
            'total_corretagem': 0.0,
            'total_ir': 0.0,
            'total_custos': 0.0,
            'n_rebalanceamentos': 0,
        }
    else:
        valores, custos_info = _backtest_rebalanceamento(
            retornos, pesos_filtrados, valor_inicial, estrategia,
            taxa_corretagem=taxa_corretagem,
            aliquota_ir=aliquota_ir,
        )
    
    retornos_carteira = valores.pct_change().dropna()
    
    return {
        'valores': valores,
        'drawdown': _calcular_drawdown_serie(valores),
        'retorno_anualizado': retorno_anualizado(retornos_carteira),
        'volatilidade_anualizada': volatilidade_anualizada(retornos_carteira),
        'drawdown_maximo': _calcular_drawdown_serie(valores).min(),
        'estrategia': estrategia,
        'custos': custos_info,  # NOVO
    }
"""
Métricas de performance e risco de carteiras.

Funções puras que recebem retornos (pandas Series ou DataFrame) e calculam
métricas anualizadas. Convenção: 252 dias úteis por ano.

As funções aceitam tanto Series (1 ativo) quanto DataFrame (vários ativos)
graças à vetorização do pandas.
"""

import numpy as np
import pandas as pd

from core.config import DIAS_UTEIS_ANO

# ============================================================
# RETORNO
# ============================================================

def retorno_anualizado(retornos: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Calcula o retorno anualizado a partir de retornos diários.
    
    Fórmula: (1 + retorno_medio_diario) ** 252 - 1
    
    Args:
        retornos: Série ou DataFrame de retornos diários (em decimal, não %)
    
    Returns:
        float (se Series) ou Series (se DataFrame, um valor por coluna)
    
    Examples:
        >>> # Retorno médio diário de 0.05% → ~13% ao ano
        >>> retorno_anualizado(pd.Series([0.0005] * 252))
        0.1346...
    """
    retorno_medio_diario = retornos.mean()
    return (1 + retorno_medio_diario) ** DIAS_UTEIS_ANO - 1

# ============================================================
# VOLATILIDADE (RISCO)
# ============================================================

def volatilidade_anualizada(retornos: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Calcula a volatilidade anualizada (desvio padrão dos retornos).
    
    Fórmula: desvio_padrao_diario * sqrt(252)
    
    Args:
        retornos: Série ou DataFrame de retornos diários
    
    Returns:
        float (se Series) ou Series (se DataFrame)
    """
    desvio_padrao_diario = retornos.std()
    return desvio_padrao_diario * np.sqrt(DIAS_UTEIS_ANO)

# ============================================================
# SHARPE RATIO
# ============================================================

def sharpe_ratio(
    retornos: pd.Series | pd.DataFrame,
    taxa_livre_anual: float,
) -> float | pd.Series:
    """
    Calcula o Sharpe Ratio anualizado.
    
    Fórmula: (retorno_anualizado - taxa_livre_anual) / volatilidade_anualizada
    
    Args:
        retornos: Retornos diários (Series ou DataFrame)
        taxa_livre_anual: Taxa livre de risco já anualizada (ex: CDI 0.13 = 13% a.a.)
    
    Returns:
        float ou Series com o Sharpe de cada coluna.
    
    Examples:
        >>> # Carteira com 15% a.a., vol 20% a.a., CDI a 10% a.a.
        >>> # Sharpe = (0.15 - 0.10) / 0.20 = 0.25
    """
    retorno_anual = retorno_anualizado(retornos)
    vol_anual = volatilidade_anualizada(retornos)
    return (retorno_anual - taxa_livre_anual) / vol_anual

# ============================================================
# SORTINO RATIO (Sharpe modificado: só considera downside)
# ============================================================

def sortino_ratio(
    retornos: pd.Series | pd.DataFrame,
    taxa_livre_anual: float,
) -> float | pd.Series:
    """
    Calcula o Sortino Ratio anualizado.
    
    Diferente do Sharpe, usa só a volatilidade dos retornos NEGATIVOS
    (downside deviation). Recompensa assimetria favorável.
    
    Args:
        retornos: Retornos diários
        taxa_livre_anual: Taxa livre de risco anualizada
    
    Returns:
        float ou Series com o Sortino.
    """
    retorno_anual = retorno_anualizado(retornos)
    
    # Pega só retornos negativos (substitui positivos por NaN)
    retornos_negativos = retornos.where(retornos < 0)
    
    # Desvio padrão só dos negativos, anualizado
    downside_dev_anual = retornos_negativos.std() * np.sqrt(DIAS_UTEIS_ANO)
    
    return (retorno_anual - taxa_livre_anual) / downside_dev_anual

# ============================================================
# DRAWDOWN MÁXIMO
# ============================================================

def drawdown_maximo(retornos: pd.Series | pd.DataFrame) -> float | pd.Series:
    """
    Calcula o drawdown máximo a partir de retornos diários.
    
    Drawdown = maior queda percentual do pico até um vale subsequente.
    Mede o "stress emocional" do investidor.
    
    Args:
        retornos: Retornos diários
    
    Returns:
        float (Series) ou Series (DataFrame), em decimal NEGATIVO.
        Ex: -0.25 significa drawdown máximo de 25%.
    """
    # 1. Acumula os retornos: 1 + r0, (1+r0)*(1+r1), ...
    acumulado = (1 + retornos).cumprod()
    
    # 2. Máximo histórico até cada ponto
    pico = acumulado.cummax()
    
    # 3. Drawdown a cada ponto
    drawdown = (acumulado - pico) / pico
    
    # 4. Pior drawdown = mínimo (valor mais negativo)
    return drawdown.min()
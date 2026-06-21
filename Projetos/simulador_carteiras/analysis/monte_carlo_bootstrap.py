"""
Monte Carlo via Bootstrap Histórico.

Gera trajetórias futuras reamostrando aleatoriamente os retornos diários
do histórico (com reposição). Preserva caudas gordas e assimetrias dos
dados reais. Não assume nenhuma distribuição estatística.
"""

import numpy as np
import pandas as pd

from core.config import N_SIMULACOES_CENARIOS, N_DIAS_PROJECAO


def simular_bootstrap(
    retornos_historicos: pd.DataFrame,
    n_simulacoes: int = N_SIMULACOES_CENARIOS,
    n_dias_futuro: int = N_DIAS_PROJECAO,
    seed: int | None = None,
) -> np.ndarray:
    """
    Gera trajetórias de retornos futuros via Bootstrap Histórico.
    
    Sorteia (com reposição) dias do histórico para compor as trajetórias.
    
    Args:
        retornos_historicos: DataFrame de retornos diários (índice=data, colunas=ativos)
        n_simulacoes: Quantas trajetórias gerar
        n_dias_futuro: Tamanho de cada trajetória (em dias úteis)
        seed: Semente do random pra reprodutibilidade (opcional)
    
    Returns:
        Array shape (n_simulacoes, n_dias_futuro, n_ativos)
        com retornos diários simulados.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Converte DataFrame em array NumPy: shape (n_dias_historicos, n_ativos)
    retornos_array = retornos_historicos.values
    n_dias_hist = len(retornos_array)
    n_ativos = retornos_array.shape[1]
    
    print(f"🎲 Bootstrap: gerando {n_simulacoes:,} trajetórias × {n_dias_futuro} dias × {n_ativos} ativos...")
    
    # Sorteia índices: matriz (n_simulacoes, n_dias_futuro)
    # Cada elemento é um índice válido em [0, n_dias_hist)
    indices = np.random.choice(
        n_dias_hist,
        size=(n_simulacoes, n_dias_futuro),
        replace=True,
    )
    
    # Indexa: shape (n_simulacoes, n_dias_futuro, n_ativos)
    retornos_simulados = retornos_array[indices]
    
    print(f"   ✅ Shape final: {retornos_simulados.shape}")
    
    return retornos_simulados
"""
Monte Carlo via Distribuição Normal Multivariada.

Assume que os retornos seguem distribuição normal com média e covariância
iguais às do histórico. Gera trajetórias futuras a partir dessa distribuição.

LIMITAÇÃO: a distribuição normal subestima eventos extremos (caudas gordas
do mercado real). Compare com Bootstrap para ver a diferença.
"""

import numpy as np
import pandas as pd

from core.config import N_SIMULACOES_CENARIOS, N_DIAS_PROJECAO


def simular_normal(
    retornos_historicos: pd.DataFrame,
    n_simulacoes: int = N_SIMULACOES_CENARIOS,
    n_dias_futuro: int = N_DIAS_PROJECAO,
    seed: int | None = None,
) -> np.ndarray:
    """
    Gera trajetórias de retornos futuros via Normal Multivariada.
    
    Calcula média e cov dos retornos históricos, depois gera amostras
    da distribuição normal multivariada com esses parâmetros.
    
    Args:
        retornos_historicos: DataFrame de retornos diários
        n_simulacoes: Quantas trajetórias gerar
        n_dias_futuro: Tamanho de cada trajetória
        seed: Semente do random (opcional)
    
    Returns:
        Array shape (n_simulacoes, n_dias_futuro, n_ativos)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_ativos = retornos_historicos.shape[1]
    
    # Estatísticas dos retornos históricos
    mu = retornos_historicos.mean().values         # vetor (n_ativos,)
    cov = retornos_historicos.cov().values         # matriz (n_ativos, n_ativos)
    
    print(f"🎲 Normal: gerando {n_simulacoes:,} trajetórias × {n_dias_futuro} dias × {n_ativos} ativos...")
    
    # Gera todas as amostras de uma vez
    # multivariate_normal aceita size com múltiplas dimensões
    retornos_simulados = np.random.multivariate_normal(
        mean=mu,
        cov=cov,
        size=(n_simulacoes, n_dias_futuro),
    )
    
    print(f"   ✅ Shape final: {retornos_simulados.shape}")
    
    return retornos_simulados
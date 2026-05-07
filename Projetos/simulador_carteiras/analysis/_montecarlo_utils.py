"""
Utilitários compartilhados pelos motores de Monte Carlo.

Aplica pesos a retornos simulados e calcula trajetórias acumuladas.
"""

import numpy as np


def aplicar_pesos_e_acumular(
    retornos_simulados: np.ndarray,
    pesos: np.ndarray,
    valor_inicial: float = 1.0,
) -> np.ndarray:
    """
    Aplica pesos da carteira aos retornos simulados e gera trajetórias acumuladas.
    
    Args:
        retornos_simulados: Array (n_simulacoes, n_dias, n_ativos)
        pesos: Vetor (n_ativos,) com pesos da carteira (soma = 1)
        valor_inicial: Valor inicial do portfolio (default = 1.0)
    
    Returns:
        Array shape (n_simulacoes, n_dias) com valor acumulado do portfolio
        em cada dia de cada simulação.
    """
    # 1. Aplica pesos: para cada (simulação, dia), calcula retorno do portfolio
    # Shape: (n_simulacoes, n_dias, n_ativos) @ (n_ativos,) → (n_simulacoes, n_dias)
    retornos_portfolio = retornos_simulados @ pesos
    
    # 2. Acumula: (1+r0)*(1+r1)*(1+r2)... ao longo dos dias
    # cumprod(axis=1): produto cumulativo no eixo "dias"
    fatores_acumulados = np.cumprod(1 + retornos_portfolio, axis=1)
    
    # 3. Aplica valor inicial
    trajetorias = valor_inicial * fatores_acumulados
    
    return trajetorias
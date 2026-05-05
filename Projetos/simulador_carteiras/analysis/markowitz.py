"""
Otimização de portfólio via Monte Carlo (Teoria Moderna de Markowitz).

Estratégia: gera N portfólios com pesos aleatórios, calcula retorno e
volatilidade anualizados de cada um, e identifica os "extremos" da
fronteira eficiente (mínima variância, máximo Sharpe).

Não plota gráficos - retorna dados estruturados para a camada de visualization.
"""

# ============================================================
# BLOCO 1 - Imports
# ============================================================

import numpy as np
import pandas as pd

from core.config import DIAS_UTEIS_ANO, N_SIMULACOES_MARKOWITZ
from analysis.metricas import (
    retorno_anualizado,
    volatilidade_anualizada,
    sharpe_ratio,
)

# ============================================================
# BLOCO 2 - Gerar Pesos Aleatórios
# ============================================================

def _gerar_pesos_aleatorios(n_simulacoes: int, n_ativos: int) -> np.ndarray:
    """
    Gera uma matriz de pesos aleatórios para N portfólios.
    
    Cada linha é um vetor de pesos que soma 1 (carteira totalmente alocada).
    
    Args:
        n_simulacoes: Quantos portfólios simular
        n_ativos: Quantos ativos tem na carteira
    
    Returns:
        Array shape (n_simulacoes, n_ativos), cada linha somando 1.
    """
    pesos_brutos = np.random.random((n_simulacoes, n_ativos))
    soma_por_linha = pesos_brutos.sum(axis=1, keepdims=True)
    return pesos_brutos / soma_por_linha

# ============================================================
# BLOCO 3 - CÁLCULO DE MÉTRICAS PARA UM CONJUNTO DE PORTFÓLIOS
# ============================================================

def _calcular_metricas_portfolios(
    pesos: np.ndarray,
    retornos_anuais: np.ndarray,
    matriz_cov_anual: np.ndarray,
    taxa_livre_anual: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula retorno, volatilidade e Sharpe para CADA portfólio simulado.
    
    Vetorizado: processa todos os N portfólios de uma vez.
    
    Args:
        pesos: Matriz (N, n_ativos)
        retornos_anuais: Vetor (n_ativos,) com retorno anualizado de cada ativo
        matriz_cov_anual: Matriz (n_ativos, n_ativos) de covariância anualizada
        taxa_livre_anual: Taxa livre de risco anualizada (escalar)
    
    Returns:
        Tupla com 3 arrays de tamanho N:
        - retornos_portfolios
        - volatilidades_portfolios
        - sharpes_portfolios
    """
    # Retorno: produto interno de cada linha de pesos com vetor de retornos
    # pesos (N, k) @ retornos (k,) → resultado (N,)
    retornos = pesos @ retornos_anuais
    
    # Variância: forma quadrática vetorizada
    # Para cada portfólio i: var[i] = pesos[i] @ cov @ pesos[i].T
    # Vetorização: usar einsum ou multiplicação por elementos
    variancias = np.einsum('ij,jk,ik->i', pesos, matriz_cov_anual, pesos)
    volatilidades = np.sqrt(variancias)
    
    # Sharpe vetorizado
    sharpes = (retornos - taxa_livre_anual) / volatilidades
    
    return retornos, volatilidades, sharpes

# ============================================================
# BLOCO 4 - FILTRAGEM DE JANELA VÁLIDA
# ============================================================

def _filtrar_dados_validos(precos: pd.DataFrame) -> pd.DataFrame:
    """
    Mantém apenas as linhas onde TODOS os ativos têm dado.
    
    Imprime aviso se o recorte for muito agressivo.
    
    Args:
        precos: DataFrame de preços, possivelmente com NaN no início
    
    Returns:
        DataFrame sem NaN.
    """
    n_total = len(precos)
    df_limpo = precos.dropna()
    n_validos = len(df_limpo)
    
    if n_validos < n_total:
        descartados = n_total - n_validos
        pct = descartados / n_total * 100
        print(f"⚠️  Janela de análise: {n_validos} dias úteis (descartados {descartados} dias = {pct:.1f}%)")
        print(f"   De {df_limpo.index.min().date()} até {df_limpo.index.max().date()}")
    
    if n_validos < 252:
        print(f"⚠️  ATENÇÃO: menos de 1 ano de dados ({n_validos} dias). Resultados pouco confiáveis.")
    
    return df_limpo

# ============================================================
# BLOCO 5 - FUNÇÃO PÚBLICA - CALCULA FRONTEIRA EFICIENTE
# ============================================================

def calcular_fronteira_eficiente(
    precos: pd.DataFrame,
    taxa_livre_anual: float,
    n_simulacoes: int = N_SIMULACOES_MARKOWITZ,
) -> dict:
    """
    Calcula a fronteira eficiente via Monte Carlo de pesos aleatórios.
    
    Args:
        precos: DataFrame com preços/níveis dos ativos (índice = data)
        taxa_livre_anual: Taxa livre de risco anualizada (ex: CDI 0.085 = 8.5%)
        n_simulacoes: Quantos portfólios simular (default: do config)
    
    Returns:
        Dicionário com:
        - 'portfolios': DataFrame (n_simulacoes × [retorno, volatilidade, sharpe, pesos...])
        - 'max_sharpe': dict com pesos e métricas do portfolio de máx Sharpe
        - 'min_variancia': dict com pesos e métricas do portfolio de mín variância
        - 'ativos': lista dos tickers (mesma ordem dos pesos)
        - 'janela': dict com data_inicio e data_fim usados
    """
    print("=" * 60)
    print(f"🎯 OTIMIZAÇÃO DE PORTFÓLIO (MARKOWITZ MONTE CARLO)")
    print("=" * 60)
    
    # 1. Filtrar janela onde todos os ativos têm dados
    precos_validos = _filtrar_dados_validos(precos)
    
    # 2. Calcular retornos diários
    retornos_diarios = precos_validos.pct_change().dropna()
    
    # 3. Anualizar retornos e covariância
    retornos_anuais = retorno_anualizado(retornos_diarios).values  # vetor
    matriz_cov_anual = retornos_diarios.cov().values * DIAS_UTEIS_ANO  # matriz
    
    ativos = retornos_diarios.columns.tolist()
    n_ativos = len(ativos)
    
    print(f"\n📊 Configuração:")
    print(f"   Ativos: {n_ativos} ({ativos})")
    print(f"   Simulações: {n_simulacoes:,}")
    print(f"   Taxa livre de risco: {taxa_livre_anual:.2%}")
    
    # 4. Gerar pesos aleatórios
    print(f"\n🎲 Gerando {n_simulacoes:,} portfólios aleatórios...")
    pesos = _gerar_pesos_aleatorios(n_simulacoes, n_ativos)
    
    # 5. Calcular métricas vetorizadamente
    print(f"📐 Calculando métricas...")
    retornos, volatilidades, sharpes = _calcular_metricas_portfolios(
        pesos=pesos,
        retornos_anuais=retornos_anuais,
        matriz_cov_anual=matriz_cov_anual,
        taxa_livre_anual=taxa_livre_anual,
    )
    
    # 6. Montar DataFrame com tudo (pesos + métricas)
    df_portfolios = pd.DataFrame({
        'retorno': retornos,
        'volatilidade': volatilidades,
        'sharpe': sharpes,
    })
    
    # Adiciona uma coluna por ativo com seu peso
    for i, ativo in enumerate(ativos):
        df_portfolios[ativo] = pesos[:, i]
    
    # 7. Identificar portfólios ótimos
    idx_max_sharpe = df_portfolios['sharpe'].idxmax()
    idx_min_var = df_portfolios['volatilidade'].idxmin()
    
    portfolio_max_sharpe = df_portfolios.loc[idx_max_sharpe]
    portfolio_min_var = df_portfolios.loc[idx_min_var]
    
    # 8. Resultado
    resultado = {
        'portfolios': df_portfolios,
        'max_sharpe': {
            'retorno': portfolio_max_sharpe['retorno'],
            'volatilidade': portfolio_max_sharpe['volatilidade'],
            'sharpe': portfolio_max_sharpe['sharpe'],
            'pesos': dict(zip(ativos, portfolio_max_sharpe[ativos].values)),
        },
        'min_variancia': {
            'retorno': portfolio_min_var['retorno'],
            'volatilidade': portfolio_min_var['volatilidade'],
            'sharpe': portfolio_min_var['sharpe'],
            'pesos': dict(zip(ativos, portfolio_min_var[ativos].values)),
        },
        'ativos': ativos,
        'janela': {
            'inicio': precos_validos.index.min().date(),
            'fim': precos_validos.index.max().date(),
            'n_dias': len(precos_validos),
        },
    }
    
    # Print resumo
    print(f"\n✅ Otimização concluída!")
    print(f"\n🏆 Portfolio MAX SHARPE:")
    print(f"   Retorno: {resultado['max_sharpe']['retorno']:.2%}")
    print(f"   Volatil: {resultado['max_sharpe']['volatilidade']:.2%}")
    print(f"   Sharpe:  {resultado['max_sharpe']['sharpe']:.2f}")
    
    print(f"\n🛡️  Portfolio MIN VARIÂNCIA:")
    print(f"   Retorno: {resultado['min_variancia']['retorno']:.2%}")
    print(f"   Volatil: {resultado['min_variancia']['volatilidade']:.2%}")
    print(f"   Sharpe:  {resultado['min_variancia']['sharpe']:.2f}")
    
    print("=" * 60)
    
    return resultado
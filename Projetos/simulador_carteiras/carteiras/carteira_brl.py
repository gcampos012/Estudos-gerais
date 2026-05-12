"""
Carteira de ativos brasileiros (BRL).

Define a composição da carteira a ser analisada pelo simulador.
A análise (Markowitz, Monte Carlo) lê esse dicionário e processa
os ativos automaticamente, com os preços vindo do data_loader.

Para criar uma carteira nova, copie este arquivo e ajuste os campos.
"""

from datetime import date

CARTEIRA = {
    # Metadados (documentação pura, não interfere em cálculos)
    'nome': 'Carteira Brasileira Diversificada',
    'descricao': (
        'Carteira multiclasse em reais com renda fixa e renda variável.'
    ),
    'moeda': 'BRL',

    # Benchmark - usado para calcular Sharpe (não é ativo investido)
    'benchmark': 'CDI',

    # Data padrão de início do histórico (pode ser sobrescrito por quem usa)
    'data_inicio_default': date(2003, 1, 1),

    # Lista de ativos investíveis
    'ativos': [
        #─── Renda Fixa ─────────────────────────────────────────
        {
            'ticker': 'SELIC',
            'classe': 'rf_pos_fixado',
            'descricao': 'Taxa Selic (pós-fixado, proxy de Tesouro Selic/CDB DI)',
        },
        {
            'ticker': 'IMA-B',
            'classe': 'rf_inflacao_geral',
            'descricao': 'Carteira teórica de NTN-B (inflação, todos os prazos)',
        },
        {
            'ticker': 'IMA-B 5 P2',
            'classe': 'rf_inflacao_curta',
            'descricao': 'Carteira teórica de NTN-B (prazos de até 5 anos)',
        },

        #─── Renda Variável Brasil ─────────────────────────────────────────
        {
            'ticker': 'DIVO11.SA',
            'classe': 'rv_brasil_total_return',
            'descricao': 'ETF de ações brasileiras',
        },
        
        # ─── Renda Variável Internacional (em BRL) ──────────────
        {
            'ticker': 'VT',
            'classe': 'rv_global_total_return',
            'descricao': 'ETF iShares SP500 (preço USD convertido para BRL)',
        },

         # ─── Commodities (USD convertido para BRL) ──────────────
        {
            'ticker': 'GLD',
            'classe': 'commodities_ouro',
            'descricao': 'ETF de ouro (preço USD convertido para BRL)',
        },
    ],
}

def get_tickers() -> list[str]:
    """Retorna apenas a lista de tickers (ativos investíveis)."""
    return [ativo['ticker'] for ativo in CARTEIRA['ativos']]

def get_tickers_com_benchmark() -> list[str]:
    """Retorna lista de tickers + o benchmark (pra cerregar tudo de uma vez)."""
    return get_tickers() + [CARTEIRA['benchmark']]


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
            'descricao': 'NTN-B até 5 anos com controle de prazo médio (P2)',
        },
        {
            'ticker': 'IRF-M',
            'classe': 'rf_prefixado',
            'descricao': 'Carteira teórica de LTN/NTN-F (prefixado)',
        },

        #─── Renda Variável Brasil ─────────────────────────────────────────
        {
            'ticker': 'DIVO11.SA',
            'classe': 'rv_brasil_dividendos',
            'descricao': 'ETF de ações brasileiras pagadoras de dividendos',
        },
        
        # ─── Renda Variável Internacional (em BRL) ──────────────
        {
            'ticker': 'IVVB11.SA',
            'classe': 'rv_eua_sp500',
            'descricao': 'ETF S&P 500 com hedge cambial (negociado em BRL)',
        },
        {
            'ticker': 'WRLD11.SA',
            'classe': 'rv_global_msci',
            'descricao': 'ETF MSCI World (negociado em BRL)',
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


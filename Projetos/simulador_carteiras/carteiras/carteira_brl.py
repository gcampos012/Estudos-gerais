"""
Carregamento de carteiras a partir de arquivos JSON.

As carteiras são definidas em arquivos JSON na pasta carteiras/configs/.
Este módulo provê funções pra carregar e validar essas configurações.

Exemplo de uso:
    from carteiras.carteira_brl import load_carteira
    
    carteira = load_carteira("carteiras/configs/balanceada.json")
    ativos = carteira['ativos']
    benchmark = carteira['benchmark']
"""

import json
from datetime import date
from pathlib import Path


# Campos obrigatórios em todo JSON de carteira
CAMPOS_OBRIGATORIOS = {'nome', 'benchmark', 'data_inicio_default', 'ativos'}


def load_carteira(path: str | Path) -> dict:
    """
    Carrega uma carteira a partir de um arquivo JSON.
    
    Args:
        path: Caminho do arquivo JSON (relativo ou absoluto).
    
    Returns:
        Dict com a configuração da carteira:
        {
            'nome': str,
            'descricao': str (opcional),
            'benchmark': str,
            'data_inicio_default': date,   # convertido de string ISO
            'ativos': list[str],
        }
    
    Raises:
        FileNotFoundError: Se o arquivo não existe.
        ValueError: Se o JSON está mal formado ou faltam campos obrigatórios.
    """
    path = Path(path)
    
    # ============================================================
    # 1. VALIDA EXISTÊNCIA
    # ============================================================
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de carteira não encontrado: {path}\n"
            f"   Carteiras disponíveis em: carteiras/configs/"
        )
    
    # ============================================================
    # 2. LÊ E PARSEIA
    # ============================================================
    with open(path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON inválido em {path}: {e}"
            ) from e
    
    # ============================================================
    # 3. VALIDA CAMPOS OBRIGATÓRIOS
    # ============================================================
    campos_presentes = set(config.keys())
    campos_faltando = CAMPOS_OBRIGATORIOS - campos_presentes
    
    if campos_faltando:
        raise ValueError(
            f"Campos obrigatórios faltando em {path.name}: {campos_faltando}\n"
            f"   Esperado: {CAMPOS_OBRIGATORIOS}"
        )
    
    # ============================================================
    # 4. VALIDA TIPOS
    # ============================================================
    if not isinstance(config['ativos'], list):
        raise ValueError(f"'ativos' deve ser lista, não {type(config['ativos']).__name__}")
    
    if len(config['ativos']) == 0:
        raise ValueError(f"'ativos' não pode ser lista vazia em {path.name}")
    
    if not all(isinstance(a, str) for a in config['ativos']):
        raise ValueError(f"Todos os itens de 'ativos' devem ser strings")
    
    # ============================================================
    # 5. CONVERTE data_inicio_default DE STRING PRA date
    # ============================================================
    try:
        data_str = config['data_inicio_default']
        config['data_inicio_default'] = date.fromisoformat(data_str)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"'data_inicio_default' deve ser string ISO (YYYY-MM-DD), "
            f"recebido: {config.get('data_inicio_default')!r}"
        ) from e
    
    return config


def get_tickers(carteira: dict) -> list[str]:
    """
    Retorna a lista de tickers (ativos) da carteira.
    
    Função utilitária pra manter compatibilidade com código existente.
    
    Args:
        carteira: Dict retornado por load_carteira()
    
    Returns:
        Lista de tickers da carteira.
    """
    return list(carteira['ativos'])
# Processamento e Padronização dos dados baixados.

# Pacotes
import pandas as pd
from pathlib import Path

def processar_xls(caminho_xls: Path) -> pd.DataFrame:
    """
    Lê o XLS baixado da Anbima Data e padroniza as colunas.
    """

    df = pd.read_excel(caminho_xls)



# IRF-M: LTNs e NTN-Fs 
# IMAB: NTNB principal e NTNBs com cupons

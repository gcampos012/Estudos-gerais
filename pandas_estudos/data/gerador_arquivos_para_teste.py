# %%
import pandas as pd
import os
print(os.getcwd())

# %%
# Janeiro 2025
movimentacoes_jan = pd.DataFrame({
    "data": pd.to_datetime([
        "2025-01-03", "2025-01-08", "2025-01-15", "2025-01-22", "2025-01-29"
    ]),
    "id_cliente": [101, 102, 104, 101, 105],
    "operacao": ["Compra", "Compra", "Venda", "Compra", "Compra"],
    "ativo": ["PETR4", "CDB Santander", "Tesouro Selic 2027", "VALE3", "LCI Bradesco"],
    "quantidade": [200, 1, 1, 150, 1],
    "valor_unitario": [38.50, 50000.00, 12450.30, 65.20, 30000.00],
})

# Fevereiro 2025
movimentacoes_fev = pd.DataFrame({
    "data": pd.to_datetime([
        "2025-02-04", "2025-02-11", "2025-02-18", "2025-02-25"
    ]),
    "id_cliente": [103, 102, 107, 101],
    "operacao": ["Compra", "Compra", "Venda", "Venda"],
    "ativo": ["HGLG11", "Tesouro IPCA+ 2035", "PETR4", "ITUB4"],
    "quantidade": [80, 1, 100, 300],
    "valor_unitario": [165.40, 3850.75, 39.80, 28.10],
})

# Março 2025
movimentacoes_mar = pd.DataFrame({
    "data": pd.to_datetime([
        "2025-03-05", "2025-03-12", "2025-03-19", "2025-03-26", "2025-03-31"
    ]),
    "id_cliente": [104, 105, 102, 107, 103],
    "operacao": ["Compra", "Compra", "Venda", "Compra", "Compra"],
    "ativo": ["Debênture VALE", "CDB Santander", "CDB Santander", "MGLU3", "BBDC4"],
    "quantidade": [1, 1, 1, 500, 200],
    "valor_unitario": [25000.00, 30000.00, 50125.40, 8.50, 14.95],
})

# Salva como CSV
movimentacoes_jan.to_csv("movimentacoes_2025_01.csv", index=False)
movimentacoes_fev.to_csv("movimentacoes_2025_02.csv", index=False)
movimentacoes_mar.to_csv("movimentacoes_2025_03.csv", index=False)

print("Arquivos criados!")
print(f"Jan: {len(movimentacoes_jan)} linhas")
print(f"Fev: {len(movimentacoes_fev)} linhas")
print(f"Mar: {len(movimentacoes_mar)} linhas")
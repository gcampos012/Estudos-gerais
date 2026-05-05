"""Debug: investigar o CDI."""

from datetime import date
from core.data_loader import carregar_precos

precos = carregar_precos(
    tickers=['CDI'],
    data_inicio=date(2020, 1, 1),
    data_fim=date(2024, 12, 31),
)

print("\n=== INVESTIGAÇÃO DO CDI ===\n")

print("Primeiras 5 linhas:")
print(precos.head())

print("\nÚltimas 5 linhas:")
print(precos.tail())

print(f"\nValor inicial: {precos['CDI'].iloc[0]:.6f}")
print(f"Valor final:   {precos['CDI'].iloc[-1]:.6f}")

retorno_total = precos['CDI'].iloc[-1] / precos['CDI'].iloc[0] - 1
print(f"\nRetorno acumulado total: {retorno_total:.4%}")

n_anos = len(precos) / 252
print(f"Número de anos: {n_anos:.2f}")

cdi_anual_geometrico = (1 + retorno_total) ** (1/n_anos) - 1
print(f"CDI anualizado (geométrico): {cdi_anual_geometrico:.2%}")
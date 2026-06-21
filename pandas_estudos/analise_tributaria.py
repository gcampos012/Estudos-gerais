# %% Imports
import pandas as pd
import os
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
import datetime as dt

# %% DADOS FIXOS
# Dataframe com os feriados nacionais
feriados = pd.read_excel("data/feriados_nacionais.xlsx")
feriados = feriados.dropna()
feriados["Data"] = pd.to_datetime(feriados["Data"], dayfirst=True)
# Variáveis fixas do projeto
BR_BUSINESS_DAYS = CustomBusinessDay(holidays=feriados["Data"])
HOJE = dt.datetime.today()
N_DIAS = 2521
RENTABILIDADE_ESTIMADA = 0.12
FATOR_DIARIO = ((1 + RENTABILIDADE_ESTIMADA) ** (1/252))
VALOR_INICIAL = 100000.00
ALIQUOTA_CC = 0.15
VALOR_COTA_INICIAL = 10.00

# %% Criando o dataframe base 
datas_projetadas = pd.bdate_range(
    start=HOJE,
    periods=N_DIAS,
    freq=BR_BUSINESS_DAYS,
)
df = pd.DataFrame({"Data": datas_projetadas})
df = df.set_index("Data")
df["Dias_corridos"] = (df.index - df.index[0]).days
df["Alíquota"] = np.select(
    [df["Dias_corridos"] <= 180,
     df["Dias_corridos"] <= 360,
     df["Dias_corridos"] <= 720],
     [0.225, 0.20, 0.175],
     default=0.15
)
df = df.reset_index()
df["Month"] = df["Data"].dt.month_name()
df["Last_day"] = df["Data"].dt.is_month_end
df.head()

# %% Colunas para cálculo do come-cotas
df["come_cotas"] = (df["Last_day"] & df["Month"].isin(["May", "November"]))
df["IR_CC"] = 0.0
df["Saldo_Atual"] = 0.0
df["Custo_Aq_Cota"] = 0.0
df["VPC"] = 0.0
df["Qntd_Cotas"] = 0.0
saldo_atual = VALOR_INICIAL
custo_aq_cc = VALOR_COTA_INICIAL
vpc_atual = VALOR_COTA_INICIAL
cotas_atuais = VALOR_INICIAL / VALOR_COTA_INICIAL 
parcela_tributada = 0.0

# %% Loop para calcular os Saldos Líquidos após o come-cotas
for i, idx in enumerate(df.index):
    if i == 0:
        df.loc[idx, "Saldo_Atual"] = saldo_atual
        df.loc[idx, "Custo_Aq_Cota"] = custo_aq_cc
        df.loc[idx, "VPC"] =  vpc_atual
        df.loc[idx, "Qntd_Cotas"] = cotas_atuais
        continue

    saldo_atual = saldo_atual * FATOR_DIARIO
    vpc_atual = vpc_atual * FATOR_DIARIO

    if df.loc[idx, "come_cotas"]:
        rendimento_periodo = vpc_atual - custo_aq_cc
        base_calculo_ir = rendimento_periodo * cotas_atuais

        if rendimento_periodo > 0: 
            ir = base_calculo_ir * ALIQUOTA_CC
            df.loc[idx, "IR_CC"] = ir
            cotas_subtraidas = ir / vpc_atual
            cotas_atuais = cotas_atuais - cotas_subtraidas
            saldo_atual = saldo_atual - ir
            custo_aq_cc += rendimento_periodo
            parcela_tributada += base_calculo_ir 

    df.loc[idx, "Saldo_Atual"] = saldo_atual
    df.loc[idx, "Custo_Aq_Cota"] = custo_aq_cc
    df.loc[idx, "VPC"] = vpc_atual
    df.loc[idx, "Qntd_Cotas"] = cotas_atuais

    if i == N_DIAS - 1:
        # Calculo da alíquota complementar (que complementa o faltante do come-cotas)
        aliquota_complementar = df.loc[idx, "Alíquota"] - ALIQUOTA_CC
        ir_complementar = parcela_tributada * aliquota_complementar
        cotas_subtraidas = ir_complementar / vpc_atual
        cotas_atuais = cotas_atuais - cotas_subtraidas

        # Calculo do IR devido desde o último come-cotas
        rendimento_periodo = vpc_atual - custo_aq_cc
        base_calculo_ir = rendimento_periodo * cotas_atuais
        ir = base_calculo_ir * df.loc[idx, "Alíquota"]
        cotas_subtraidas = ir / vpc_atual
        cotas_atuais = cotas_atuais - cotas_subtraidas
        saldo_atual = saldo_atual - ir_complementar - ir
        custo_aq_cc += rendimento_periodo

        df.loc[idx, "Saldo_Atual"] = saldo_atual
        df.loc[idx, "Custo_Aq_Cota"] = custo_aq_cc
        df.loc[idx, "VPC"] = vpc_atual
        df.loc[idx, "Qntd_Cotas"] = cotas_atuais

#idx_evento = df.index[df["come_cotas"]][1]
#print(df.iloc[idx_evento-2:idx_evento+3])
print(f"\n{df.tail()}")

#with pd.option_context("display.max_rows", None):
#    print(df)
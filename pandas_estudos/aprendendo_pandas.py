# %%
import pandas as pd
import os
import numpy as np

tabela = pd.read_csv("data/precotaxatesourodireto.csv", sep=";")

# %% Series
titulos = pd.Series(tabela["Tipo Titulo"])
type(titulos)

# %% DataFrames
titulo_e_taxa = pd.DataFrame(tabela[["Tipo Titulo", "Taxa Compra Manha"]])
type(titulo_e_taxa)


# %% =======================================================================
# 3) NAVEGANDO PELOS DADOS
# ==========================================================================


# %% Informações Básicas
titulo_e_taxa.head(10) # .head() método
titulo_e_taxa.tail(10) # .tail() método
titulo_e_taxa.sample(10) # .sample() método
titulo_e_taxa.shape # .shape é um atributo 
titulo_e_taxa.columns # .columns é um atributo
titulo_e_taxa.index # .index é um atributo
titulo_e_taxa.info(memory_usage='deep')  
titulo_e_taxa.dtypes # .dtypes é um atributo 
# .dtypes mostra os valores da TIPAGEM de cada campo (colunas)titulo_e_taxa

# %% Renomeação de colunas
"""
    .rename() -> função que irá renomear o que passarmos dentro
    columns = {} -> a função coluna nos diz que iremos renomear as colunas, e passamos um dict 
    dentro, sendo a chave a coluna ANTIGA, e o valor a coluna NOVA
"""
# %% Renomeando colunas. Atenção: estamos apenas criando um NOVO dataframe renomeado
titulo_e_taxa.rename(columns={"Taxa Compra Manha": "Tx_Compra"})

# %% Renomeando colunas e reatribuindo ao próprio dataframe
titulo_e_taxa = titulo_e_taxa.rename(columns={"Taxa Compra Manha": "Tx_Compra"})

# %% Renomear colunas sem reatribuir, apenas com inplace = True
titulo_e_taxa.rename(columns={"Taxa Compra Manha": "Tx_Compra"}, inplace=True)
titulo_e_taxa

# %% Nesse formato, estamos selecionando apenas a SERIE Tx_Compra
titulo_e_taxa["Tx_Compra"]

#%% Nesse formato, mantemos a configuração de dataframe
titulo_e_taxa[["Tx_Compra"]]
 
# %% Nesse formato, mantemos as colunas "titulo titulo" e "tx_compra", formato dataframe 
titulo_e_taxa[["Tipo Titulo", "Tx_Compra"]]

# %% Ordenar colunas em forma alfabética
colunas = titulo_e_taxa.columns.to_list()
colunas.sort()
colunas

# %% Reordenar colunas com base numa lista contendo as colunas 
titulo_e_taxa = titulo_e_taxa[colunas]
titulo_e_taxa


# %%=====================================================================
# 4) FILTRANDO DADOS E OPERADORES DE FILTRAGEM
# =======================================================================


# %% Tratanto a base 
df = tabela
df["Taxa Compra Manha"] = df["Taxa Compra Manha"].str.replace(",",".").astype(float)
df["Taxa Venda Manha"] = df["Taxa Venda Manha"].str.replace(",",".").astype(float)

# %%
df.info()

# %% Criando um filtro "e" e aplicando o filtro no dataframe
maior_6 = (df["Taxa Compra Manha"] >= 6) & (df["Tipo Titulo"] == "Tesouro IPCA+") & (df["Data Vencimento"] == "15/05/2015") 
df[maior_6]
# maior_6 é uma série com true e false em todas as linhas da serie
# essa série tem a mesma quantidade de elementos que a quantidade total de linhas
# só podemos fazer isso pois ambos tem a mesma dimensão em linhas

# %% Filtro "ou", ou seja, pmr >= 730 ou <= 2000
maior_7 = (df["Taxa Compra Manha"] > 6) | (df["Taxa Compra Manha"] <= 12) 
df[maior_7]

# %% método isin(): serve para verificar se algum valor está contido dentro de uma lista/iterável 
filtro = df["Taxa Compra Manha"].isin([6,7.10])
df[filtro]

# %% verificar se existe algum NA
filtro_na = df["Taxa Compra Manha"].isna()
filtro_na

# %% fazendo a negação do método usando o tíu (~) antes
filtro_na = ~df["Taxa Compra Manha"].isna()
filtro_na

# %% Filtrando dados específicos
taxa_maior_13 = df["Taxa Compra Manha"] >= 18
df[taxa_maior_13]

"""
    Filtro NÃO é cópia, ou seja, quando estamos filtrando não estamos criando
    uma cópia do dataframe, mas sim criando uma view onde somente os dados =
    true aparecem.

    Se quisermos fazer uma cópia de um dataframe, basta colocar ao fim .copy()
"""


# %% ====================================================================
# 5) TRANSFORMAÇÕES E REMOÇÕES
# =======================================================================

df.info()
df.head(5)

# %% Transformando dados str em datetime
df["Data Vencimento"] = pd.to_datetime(df["Data Vencimento"])
df["Data Base"] = pd.to_datetime(df["Data Base"])

# %% Transformando dados str em float
df["PU Compra Manha"] = df["PU Compra Manha"].str.replace(",", ".").astype(float)
df["PU Venda Manha"] = df["PU Venda Manha"].str.replace(",", ".").astype(float)
df["PU Base Manha"] = df["PU Base Manha"].str.replace(",", ".").astype(float)

# %% Vendo como os dados estão
df.info()

# %% Criando coluna "spreads" e ordenando 
df["spread"] = df["Taxa Venda Manha"] - df["Taxa Compra Manha"]
df.sort_values(by=["Tipo Titulo", "Taxa Compra Manha"], ascending=[True, False]).tail()

# %% Checando ano e mes usando método dt.
df["Data Vencimento"].dt.year
df["Data Vencimento"].dt.month_name()

# %% lendo arquivo xls
idaipca = pd.read_excel("IDAIPCAINFRAESTRUTURA-HISTORICO.xls")
idaipca

# %% view sem os NaN
idaipca.dropna(how="any", subset="Variação no Mês (%)")
# how="all" -> se todos forem NaN, ai sim remove
# how="any" -> se ao menos UM for NaN, já remove
# subset="nomedacoluna" -> é o subset que usaremos para "filtrar" o dropna  

# %% substituindo NaN por alguma coisa
idaipca["Variação no Mês (%)"].fillna(0) # aqui substituimos os NaN da coluna Variação mes
idaipca.fillna(0) # aqui estamos substituindo todos os NaNs do dataframe

# %% Removendo duplicatas 
listagem = df.drop_duplicates(subset=["Data Vencimento","Tipo Titulo"]).sort_values(by="Data Vencimento")
listagem

# %% Mantendo a primeira taxa disponível de cada título individualmente 
df = df.sort_values(by="Data Base")  
primeira_taxa = df.drop_duplicates(keep='first', subset=["Tipo Titulo", "Taxa Compra Manha"])
primeira_taxa

# %% Pegando outra planilha para trabalhar
df_feriados = pd.read_excel("feriados_nacionais.xlsx")
df_feriados = df.dropna()
df_feriados.head()

# %% Criando função para separar string e pegar a última palavra apenas
def get_last_name(x):
    return x.split(" ")[-1]

# %% Mostrando o jeito INEFICIENTE de fazer uma alteração de dados em DFs

ultimo_nome = []

for i in df_feriados["Feriado"]:
    ultimo = get_last_name(i)
    ultimo_nome.append(ultimo)

df["Ultimo_Nome"] = ultimo_nome
df.head(5)

# %% Utilizando método .apply para executar de forma mais eficiente
df["Feriado"].apply(get_last_name)
# UMA LINHA APENAS!!!!!!!!!!


# %% ====================================================================
# 5) Método Apply e GroupBy
# =======================================================================

# %% Editando formatods com apply 
dfs_aula_apply = pd.read_csv("data/precotaxatesourodireto.csv", sep=";")
dfs_aula_apply.dtypes

# função para alterar valores str para float
def str_to_float(x:str):
    x = (x.replace(",", "."))
    return float(x)

dfs_aula_apply.info()

# %% Usando método apply
dfs_aula_apply["Taxa Compra Manha"] = dfs_aula_apply["Taxa Compra Manha"].apply(str_to_float)
dfs_aula_apply["Taxa Venda Manha"] = dfs_aula_apply["Taxa Venda Manha"].apply(str_to_float)
dfs_aula_apply["PU Compra Manha"] = dfs_aula_apply["PU Compra Manha"].apply(str_to_float)
dfs_aula_apply["PU Venda Manha"] = dfs_aula_apply["PU Venda Manha"].apply(str_to_float)
dfs_aula_apply["PU Base Manha"] = dfs_aula_apply["PU Base Manha"].apply(str_to_float)

# O método apply consegue aplicar transformações nos dados das colunas e também 
# nos dados das LINHAS. Quando utilizamos axis=1, estamos falando "utilize as linhas"

# %% Usando filtro e aplicando a média da taxa de compra manha
filtro = (dfs_aula_apply["Data Vencimento"] == "15/08/2024") & (dfs_aula_apply["Tipo Titulo"] == "Tesouro IPCA+")
dfs_aula_apply[filtro]["Taxa Compra Manha"].mean()

# %% Média das taxas de compra e venda
taxas = ["Taxa Compra Manha", "Taxa Venda Manha"]
dfs_aula_apply[filtro][["Taxa Compra Manha", "Taxa Venda Manha"]].mean()

# %% Dica para encontrar quais colunas contém valores do tipo str e criando uma lista depois
num_columns = dfs_aula_apply.dtypes[~(dfs_aula_apply.dtypes == "str")].index.tolist()
# O codigo acima usamos apenas para encontrat quais colunas sao do tipo numericas

dfs_aula_apply[filtro][num_columns].describe()

# %% Formatando em datetime
dfs_aula_apply["Data Vencimento"] = pd.to_datetime(dfs_aula_apply["Data Vencimento"])
dfs_aula_apply["Data Base"] = pd.to_datetime(dfs_aula_apply["Data Base"])

# %% pegando a média das taxas por tipo de titulo
dfs_aula_apply.groupby(by=["Tipo Titulo"], as_index=False)[["Taxa Compra Manha"]].mean()

# %% 
dfs_aula_apply.head()
summary = (dfs_aula_apply.groupby(by=["Tipo Titulo"], as_index=False)
                         .agg({"Data Base": ['count'],
                               "Taxa Compra Manha": ['mean', 'median']}))

summary
# quando fazemos desta forma, summary se torna um multindex, que acaba sendo ruim de trabalhar 
# pois define uma hierarquia entre os dados

# %% Acessando uma série de uma agregação de forma mais eficiente 
summary.columns = ["Tipo Titulo", 'qtd_dados', 'Media_Tx_Compra', 'Mediana_Tx_Compra']
summary

# %%
def diff(x: pd.Series):
    amplitude = x.max() - x.min()
    media = x.mean()
    return np.sqrt((amplitude - media)**2)

# %%

(dfs_aula_apply.groupby(by=["Tipo Titulo"])  
               .agg({
                   "Data Base": ['count'],
                   "Taxa Compra Manha": ["mean", "median", diff]
               })
)

# %% 
def deltaTaxas(x: pd.Series):
    media = x.mean()
    mediana = x.median()
    deltatx = media - mediana
    return deltatx

(dfs_aula_apply.groupby(by=["Tipo Titulo"])
               .agg({
                   "Taxa Compra Manha": ["mean", "median", deltaTaxas]  
               }))

# Dentro do groupby podemos criar funções/métodos MUITO personalizáveis

# %% ====================================================================
# Cruzamento de dados utilizando Merge
# =======================================================================

# Base 1: Cadastro de clientes
clientes = pd.DataFrame({
    "id_cliente": [101, 102, 103, 104, 105, 106, 107],
    "nome": ["Ana Silva", "Bruno Costa", "Carla Mendes", "Diego Souza", 
             "Eduarda Lima", "Fábio Ramos", "Gabriela Nunes"],
    "perfil": ["Conservador", "Balanceado", "Conservador", "Arrojado", 
               "Balanceado", "Conservador", "Arrojado"],
    "patrimonio": [150000, 480000, 220000, 1200000, 350000, 95000, 2100000],
    "cidade": ["Sorocaba", "São Paulo", "Sorocaba", "Campinas", 
               "São Paulo", "Sorocaba", "Campinas"],
})

# Base 2: Aplicações dos clientes (alguns clientes têm várias aplicações, outros não aparecem)
aplicacoes = pd.DataFrame({
    "id_aplicacao": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "id_cliente": [101, 101, 102, 103, 104, 104, 104, 105, 107, 999],
    "produto": ["CDB", "Tesouro Selic", "LCI", "CDB", "Debênture Incentivada",
                "Fundo Multimercado", "Ações", "LCA", "Fundo Imobiliário", "CDB"],
    "valor_aplicado": [50000, 30000, 200000, 100000, 300000, 
                       400000, 500000, 150000, 800000, 25000],
    "data_aplicacao": pd.to_datetime([
        "2024-03-15", "2024-05-20", "2024-01-10", "2024-07-08", "2024-02-25",
        "2024-06-12", "2024-09-30", "2024-04-18", "2024-11-05", "2025-01-22"
    ]),
})

# Base 3: Aplicações novas de 2025 (mesma estrutura que aplicacoes, para usar com concat)
aplicacoes_2025 = pd.DataFrame({
    "id_aplicacao": [11, 12, 13, 14, 15],
    "id_cliente": [102, 103, 106, 107, 108],
    "produto": ["Tesouro IPCA+", "CDB", "Poupança", "Ações", "Fundo de Renda Fixa"],
    "valor_aplicado": [80000, 75000, 15000, 250000, 60000],
    "data_aplicacao": pd.to_datetime([
        "2025-02-14", "2025-03-08", "2025-01-30", "2025-04-12", "2025-05-25"
    ]),
})

print("=== CLIENTES ===")
print(clientes.head())
print(f"\n=== APLICAÇÕES ===")
print(aplicacoes.head())
print(f"\n=== APLICAÇÕES 2025 ===")
print(aplicacoes_2025.head())

# %% Aprendendo a juntar duas tabelas com Pandas
aplicacoes.merge( # aplicacoes será a base da esquerda, ou seja, partimos de aplicacoes
    right=clientes, # clientes será a base da direita, ou seja, a base de "procura"
    how='left', # o how define COMO iremos fazer o join
    on=["id_cliente"] # aqui colocamos qual coluna conterá o dado fundamental em ambas as tabelas
)

# how = 'inner' -> considera apenas as linhas contidas em AMBAS as tabelas/dataframes

# how = 'left' -> considera os dados da tabela da ESQUERDA, ou seja, ela ficará fixa
# e procuramos os dados da outra base, caso não tenha nada nessa outra base, ficará NaN

# how = 'right' -> considera os dados da tabela da DIREITA, ou seja, a direita ficará
# fixa e caso não tenha dados na base da esquerda, os dados serão inputados como NaN


# %% Se tivermos de deixar explícita quais colunas serão as primary keys, fazemos assim:
aplicacoes.merge( 
    right=clientes,
    left_on=["id_cliente"],
    right_on=["id_cliente"],
    how='left', 
)
# Dicas:
# 1) Nas tabelas a direita, filtre os dados para apenas aquilo que queira, isso torna mais performatico

# %% ====================================================================
# Utilizando concat
# =======================================================================

def read_file(file_name:str):
    df = (pd.read_csv(f"data/arquivos_csv/{file_name}.csv", sep=",")
          .rename(columns={"valor_unitario":file_name})
          .set_index(["data", "id_cliente"]))
    return df

# %%
file_names = os.listdir("data/arquivos_csv")

dfs = []
for i in file_names:
    file_name = i.split(".")[0]
    dfs.append(read_file(file_name))

df_full = (pd.concat(dfs, axis=1)
           )

df_full

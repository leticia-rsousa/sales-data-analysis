# Importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando o estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# --- Geração de Dados Fictícios Coerentes ---
print("\nGerando conjunto de dados fictícios...")
np.random.seed(42)

# Criando um dicionário de dados
data = {
    'ID_Pedido': range (1001, 1101),
    'Data_Compra': pd.to_datetime(pd.date_range(start = '2026-07-01', periods = 100, freq = 'D')) - pd.to_timedelta(np.random.randint(0, 30, size = 100), 'd'),
    'Cliente_ID': np.random.randint(100, 150, 100),
    'Produto': np.random.choice(['Smartphone', 'Notebook', 'Fone de Ouvido', 'Smartwatch', 'Teclado Mecânico'], size = 100),
    'Categoria': ['Eletrônicos', 'Eletrônicos', 'Acessórios', 'Acessórios', 'Acessórios'] * 20,
    'Quantidade': np.random.randint(1, 5, size = 100),
    'Preco_Unitario': [5999.90, 8500.00, 799.50, 2100.00, 850.00] * 20,
    'Status_Entrega': np.random.choice(['Entregue', 'Pendente', 'Cancelado'], size = 100, p = [0.8, 0.15, 0.05])
}

# Criando o dataframe a partir do dicionário
df_vendas = pd.DataFrame(data)

# --- Introduzindo Problemas nos Dados para o Exercício ---
print("\nIntroduzindo problemas nos dados para a limpeza...\n")

# Valores Ausentes (NaN)
df_vendas.loc[5:10, 'Quantidade'] = np.nan
df_vendas.loc[20:22, 'Status_Entrega'] = np.nan
df_vendas.loc[30, 'Cliente_ID'] = np.nan

# Dados Duplicados
df_vendas = pd.concat([df_vendas, df_vendas.head(3)], ignore_index = True)

# Tipos de Dados Incorretos
df_vendas['Preco_Unitario'] = df_vendas['Preco_Unitario'].astype(str)
df_vendas.loc[15, 'Preco_Unitario'] = 'valor_invalido'  # Simulando um erro de digitação
df_vendas['Cliente_ID'] = df_vendas['Cliente_ID'].astype(str)

# Outliers
df_vendas.loc[50, 'Quantidade'] = 50  # Um valor claramente fora do padrão

print("Dados gerados com sucesso!\n")

# Visualizando primeiras e últimas linhas
print(df_vendas.head())
print(df_vendas.tail())

# Informações gerais do DataFrame
print("\n--- Informações Gerais do DataFrame ---\n")
print(df_vendas.info())

# Valores ausentes
print("\n--- Verificando Valores Ausentes ---\n")
print(df_vendas.isna().sum())

# Linhas duplicadas
print("\n--- Verificando a Presença de Registros Duplicados ---\n")
print(f"Número de linhas duplicadas: {df_vendas.duplicated().sum()}")

# Estatísticas descritivas
print("\n--- Estatísticas Descritivas para Colunas Numéricas ---\n")
print(df_vendas.describe()) # Preco_Unitario não aparece por ser 'object'

print("\n--- Estatísticas Descritivas para Colunas Categóricas ---\n")
print(df_vendas.describe(include = ['object']))

# Tipos de dados
print("\n--- Tipo de Dados ---\n")
print(df_vendas.dtypes)

# Criando cópia para limpeza
df_limpo = df_vendas.copy()

# --- Corrigindo Tipos de Dados ---
print("Corrigindo tipos de dados...")
df_limpo['Preco_Unitario'] = pd.to_numeric(df_limpo['Preco_Unitario'], errors = 'coerce')
df_limpo['Cliente_ID'] = pd.to_numeric(df_limpo['Cliente_ID'], errors = 'coerce').astype('Int64') # Usamos Int64 para permitir NaN
print(df_limpo.dtypes)

# --- Tratamento de Valores Ausentes ---
print("Tratando valores ausentes...")

# Quantidade: preencher com mediana (mais robusta a outliers)
mediana_qtd = df_limpo['Quantidade'].median()
df_limpo.fillna({'Quantidade': mediana_qtd}, inplace = True)

# Status_Entrega: preencher com moda (valor mais frequente)
moda_status = df_limpo['Status_Entrega'].mode()[0]
df_limpo['Status_Entrega'] = df_limpo['Status_Entrega'].fillna(moda_status)

# Preco_Unitario e Cliente_ID: remover linhas com NaN gerado por erro
df_limpo.dropna(subset = ['Preco_Unitario', 'Cliente_ID'], inplace = True)

# --- Removendo Duplicatas ---
print("Removendo registros duplicados...")
df_limpo.drop_duplicates(inplace = True)

# --- Tratamento de Outliers ---
print("Tratando outliers...")
sns.boxplot(x = df_limpo['Quantidade'])
plt.title('Boxplot de Quantidade (Antes de tratar outlier)')
plt.show()

# Remover valores muito distantes da média (> 3 desvios padrão)
limite_superior = df_limpo['Quantidade'].mean() + 3 * df_limpo['Quantidade'].std()
df_limpo = df_limpo[df_limpo['Quantidade'] < limite_superior]

# Boxplot após remoção de outlier
sns.boxplot(x = df_limpo['Quantidade'])
plt.title('Boxplot de Quantidade (Depois de tratar outlier)')
plt.show()

# --- Verificação Final ---
print("\n--- Verificação Final Pós-Limpeza ---\n")
print(df_limpo.info())
print("\nValores ausentes restantes:\n", df_limpo.isna().sum())
print(f"\nLinhas duplicadas restantes: {df_limpo.duplicated().sum()}")

# --- Feature Engineering ---
df_limpo['Total_Venda'] = df_limpo['Quantidade'] * df_limpo['Preco_Unitario']

# Receita total
receita_total = df_limpo['Total_Venda'].sum()
print(f"A receita total da loja foi de: R${receita_total: .2f}")

# Receita por categoria
receita_por_categoria = df_limpo.groupby('Categoria')['Total_Venda'].sum().sort_values(ascending = False)
print("\n--- Receita Total por Categoria ---\n")
print(receita_por_categoria)

# Produto mais vendido em quantidade
produto_mais_vendido = df_limpo.groupby('Produto')['Quantidade'].sum().sort_values(ascending = False)
print("\n--- Total de Unidades Vendidas por Produto ---\n")
print(produto_mais_vendido)

# --- Análise de Vendas ao Longo do Tempo ---
vendas_por_dia = df_limpo.set_index('Data_Compra').resample('D')['Total_Venda'].sum()
print("\n--- Resumo de Vendas por Dia (Primeiros 5 dias) ---\n")
print(vendas_por_dia.head())

# --- Visualizações ---
# Receita por Categoria
receita_por_categoria.plot(kind = 'bar', color = 'skyblue')
plt.title('Receita Total por Categoria de Produto')
plt.ylabel('Receita (R$)')
plt.xlabel('Categoria')
plt.xticks(rotation = 0)
plt.show()

# Quantidade vendida por Produto
produto_mais_vendido.plot(kind = 'barh', color = 'salmon')
plt.title('Quantidade de Unidades Vendidas por Produto')
plt.ylabel('Produto')
plt.xlabel('Quantidade Vendida')
plt.gca().invert_yaxis() # Maior valor no topo
plt.show()

# Tendência de Vendas ao Longo do Tempo
vendas_por_dia.plot(kind = 'line', marker = '.', linestyle = '--')
plt.title('Tendência de Vendas Diárias')
plt.ylabel('Receita (R$)')
plt.xlabel('Data da Compra')
plt.grid(True)
plt.show()

# Distribuição do Status de Entrega
import plotly.express as px

distribuicao_status = px.pie(
    values = status_counts,
    names = status_counts.index,
    hole = 0,
    title = 'Distribuição do Status de Entrega',
)

distribuicao_status.update_traces(
    pull = [0.05 if i == maior_idx else 0 for i in range(len(status_counts))],
)

distribuicao_status.show()

## Sales Data Analysis
**Descrição Geral** 📄<br>
Este projeto apresenta uma **análise de dados de vendas**, utilizando **Python** e bibliotecas de manipulação e visualização de dados. O sistema gera **dados fictícios de vendas**, realiza **limpeza, tratamento de valores ausentes e outliers, criação de colunas calculadas** e produz **gráficos estáticos e interativos** para facilitar a interpretação dos resultados. O projeto demonstra conceitos de **engenharia de dados, visualização e análise exploratória de dados.**

---
**Objetivo** 🎯 <br> 
O objetivo principal do projeto é fornecer uma **ferramenta prática para manipulação e análise de dados de vendas**, abrangendo **limpeza de dados, engenharia de atributos e análise exploratória**, permitindo gerar insights sobre faturamento, produtos mais vendidos e desempenho por categoria e status de entrega.

---
**Tecnologias Utilizadas** 💻 <br>
* ***Python*** - linguagem principal.
* ***Pandas*** - manipulação de dados em DataFrames.
* ***NumPy*** - operações matemáticas e geração de números aleatórios.
* ***Matplotlib / Seaborn*** - criação de gráficos estáticos.
* ***Plotly*** - criação de gráficos interativos.

---
**Arquitetura e Estrutura do Código** 🧱 <br><br>
***1. Script Principal (sales_data_analysis.py)*** <br>
Responsável por:
* ***Gerar dados fictícios de vendas.*** 
* ***Limpar dados, tratar valores ausentes e outliers.***
* ***Criar colunas adicionais (Total_Venda, Status_Entrega).***
* ***Executar análises estatísticas básicas (shape, resumo, tipos de dados).***
* ***Agrupar dados para identificar produtos mais vendidos e faturamento por categoria e dia.***
* ***Criar gráficos estáticos e interativos para visualização dos resultados.***

---
**Conceitos e Funcionalidades Demonstradas** 🔍 <br><br>
✅ ***Manipulação de dados:*** <br>
Uso de **Pandas e NumPy** para gerar, organizar e processar os dados de vendas.

✅***Limpeza de dados:*** <br>
Tratamento de **valores ausentes, dados duplicados, tipos de dados incorretos e outliers.**

✅***Engenharia de atributos:*** <br>
Criação de colunas calculadas, como **Total_Venda e Status_Entrega**, para enriquecer a análise.

✅***Visualização de dados:*** <br>
Criação de **gráficos de barras, linha e pizza** para análise de faturamento e produtos.

✅***Agrupamento e análise:*** <br>
Identificação de **produtos mais vendidos e faturamento por categoria e por dia.**

---
**Como Executar o Projeto** ▶️ <br><br>
***1. Instale as dependências (recomendado via requirements.txt):*** <br>
```pip install -r requirements.txt```

***2. Execute o script principal:*** <br>
```python sales_data_analysis.py```

***3. Siga as instruções no terminal e veja os gráficos gerados.*** <br>

***Exemplo de saída:*** <br>
```
Dados gerados: (100, 8)

Top Produtos Mais Vendidos
Smartphone       35
Notebook         30
Fone de Ouvido   28
...

Faturamento Total: R$ 500.000,00
Gráficos de faturamento por categoria, produtos e vendas diárias são exibidos.
```

---
**Conclusão** 📌 <br>
Este projeto demonstra como realizar **limpeza de dados, engenharia de atributos e análise exploratória de vendas**, integrando **manipulação de dados, cálculos financeiros e visualizações gráficas.** Ele serve como um exemplo prático de **organização de código e exploração de informações**, permitindo gerar insights relevantes sobre o desempenho de produtos e faturamento.

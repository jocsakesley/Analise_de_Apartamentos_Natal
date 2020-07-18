import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score

#Título
st.title("Alugueis de apartamentos - Natal/RN")
st.write("A base de dados foi coletada a partir do site do Viva Real")

# Função para ler o arquivo contendo o dataset
def load_data():
    return pd.read_csv("Webapp/data_features.csv")

#Função para treinar o modelo
def train_model():
    df = load_data()
    X = df.drop(['aluguel', 'bairro', 'Unnamed: 0'], axis=1)
    y = df['aluguel']
    model_RF = RandomForestRegressor(max_features=3, max_depth=10, random_state=10)
    model_RF.fit(X,y)
    return model_RF

#Definindo a variável com o modelo treinado
model = train_model()

#Definindo a variável com os dados carregados
data = load_data()

#Página principal para análise exploratória
st.subheader("Explore a base de dados")

#Definindo colunas default para apresentação do dataset
defaultcolumns = ['bairro', 'area', 'quartos', 'aluguel']
cols = st.multiselect('Selecione os atributos desejados para conhecer a base de dados', data.drop('Unnamed: 0', axis=1).columns.tolist(), default=defaultcolumns)
st.dataframe(data[cols])

#Plotando histgrama com a opção de escolher a faixa de valores
st.subheader("Distribuição de apartamentos por valor de aluguel")
valores = st.slider("Faixa de aluguéis:", data['aluguel'].min(), data['aluguel'].max(), (500.0, 1800.0))
dados = data[data['aluguel'].between(left=valores[0], right=valores[1])]
hist = px.histogram(dados, 'aluguel', nbins=100, title="Distribuição de alugueis: ")
hist.update_xaxes(title='Aluguel')
hist.update_yaxes(title='Total')
st.plotly_chart(hist)

opc = st.selectbox("Selecione o que você quer analisar (Contagem/Média de alugueis):", ['Média', 'Contagem'] )

if opc == 'Contagem':
    bairros = data.groupby(data['bairro'])['bairro'].count()
    bairros = pd.DataFrame(bairros)
    bairros = bairros['bairro'].sort_values(ascending=False)
    bar_bairros = px.bar(bairros)
    bar_bairros.update_xaxes(title="Bairros")
    bar_bairros.update_yaxes(title="Quantidade")
else:
    bairros = data.groupby(data['bairro'])['aluguel'].mean()
    bairros = pd.DataFrame(bairros)
    bairros = bairros['aluguel'].sort_values(ascending=False)
    bar_bairros = px.bar(bairros)
    bar_bairros.update_xaxes(title="Bairros")
    bar_bairros.update_yaxes(title="Valores")

st.plotly_chart(bar_bairros)

# Criação de uma sidebar com os dados da previsão a serem preenchidos pelo usuário
st.sidebar.subheader("Selecione as características para previsão:")
# Criando uma lista ordenada com os valores únicos para os bairros
bairros = list(data['bairro'].drop_duplicates().sort_values())
bairro = st.sidebar.selectbox("Selecione o bairro: ", bairros)
#Concatenando os valores do bairro com os valores de bairro codificados em número, para entrada no modelo
tuplas_bairros = zip(data['bairro'], data['bairro_encoded'])
lista_bairros = list(tuplas_bairros)
#Substituido o valor de bairro escolhido pelo usuário para o seu respectivo código
bairro_unicos = []
for lb in lista_bairros:
    if lb not in bairro_unicos:
        bairro_unicos.append(lb)
for bu in bairro_unicos:
    if bu[0] == bairro:
        bairro = bu[1]

area = st.sidebar.slider("Selecione a área:", 40, int(data['area'].max()))

quartos = st.sidebar.number_input("Selecione o número de quartos:", value=int(data['quartos'].mean()))

banheiros = st.sidebar.number_input("Selecione o número de banheiros:", value=int(data['banheiros'].mean()))

suite = st.sidebar.number_input("Selecione o número de suítes:", value=int(data['suite'].mean()))

vagas = st.sidebar.number_input("Selecione o número de vagas:", value=int(data['vagas'].mean()))

condominio = st.sidebar.slider("Selecione o valor do condomínio:", int(data['condominio'].min()), int(data['condominio'].max()))

#Definindo um botão para a previsão do aluguel com base nas características selecionadas
button = st.sidebar.button("Prever valor do aluguel")

if button:
    result = model.predict([[bairro, area, quartos, banheiros, suite, vagas, condominio]])
    st.sidebar.subheader("O valor previsto do aluguel é: ")
    result_string = f"R$ {str(round(float(result), 2))}"
    st.sidebar.markdown(result_string)
    #bar = px.bar(data, range(len(data['aluguel'])), 'aluguel')
    #line = px.line(data, range(len(data['aluguel'])), [result for i in 'aluguel'] )
    #st.plotly_chart([bar, line])


st.write("App Desenvolvido por: Jocsã Kesley Oliveira")
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Faltas",
    page_icon="🏥",
    layout="wide"
)

# --- 1. CARREGAMENTO DE DADOS ---
@st.cache_data
def load_data():
    df = pd.read_csv('KaggleV2-May-2016.csv')

    # Renomear colunas
    df.rename(columns={
        'Hipertension': 'Hipertensao', 'Handcap': 'PCD', 'No-show': 'Faltou',
        'AppointmentDay': 'DataConsulta', 'ScheduledDay': 'DataAgendamento',
        'Neighbourhood': 'Bairro', 'Scholarship': 'BolsaFamilia', 
        'Age': 'Idade', 'Alcoholism': 'Alcoolismo', 'SMS_received': 'Recebeu_SMS'
    }, inplace=True)

    # Tratamento de Datas
    df['DataConsulta'] = pd.to_datetime(df['DataConsulta']).dt.normalize()
    df['DataAgendamento'] = pd.to_datetime(df['DataAgendamento']).dt.normalize()
    
    # Feature Engineering
    df['DiasEspera'] = (df['DataConsulta'] - df['DataAgendamento']).dt.days
    df = df[df['DiasEspera'] >= 0] 
    
    # Target Numerico
    df['Faltou'] = df['Faltou'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Tradução dos Dias da Semana
    df['DiaSemana'] = df['DataConsulta'].dt.day_name()
    traducao = {
        'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
        'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['DiaSemana'] = df['DiaSemana'].map(traducao)

    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Erro: CSV não encontrado.")
    st.stop()

# --- 2. TREINAMENTO DO MODELO ---
@st.cache_resource 
def train_model(data):
    features = ['Idade', 'BolsaFamilia', 'Hipertensao', 'Diabetes', 'Alcoolismo', 'PCD', 'Recebeu_SMS', 'DiasEspera']
    
    X = data[features]
    y = data['Faltou']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, features, acc

model, feature_names, acuracia_modelo = train_model(df)

# --- 3. BARRA LATERAL ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
st.sidebar.header("Filtros")
bairros = sorted(df['Bairro'].unique())
filtro_bairro = st.sidebar.multiselect("Bairro", bairros)

df_viz = df.copy()
if filtro_bairro:
    df_viz = df_viz[df_viz['Bairro'].isin(filtro_bairro)]

# --- 4. DASHBOARD ---
st.title("🏥 Análise de Faltas em Consultas Médicas Agendadas")
st.markdown("**Status:** Modelo Random Forest Ativo")

# KPIs
c1, c2, c3, c4 = st.columns(4)
total = len(df_viz)
faltas = df_viz['Faltou'].sum()
taxa = (faltas/total)*100 if total > 0 else 0

with c1: st.metric("Total Agendado", f"{total:,.0f}")
with c2: st.metric("Faltas Previstas", f"{faltas:,.0f}")
with c3: st.metric("Taxa de Absenteísmo", f"{taxa:.1f}%", delta_color="inverse")
with c4: st.metric("Confiabilidade da IA", f"{(acuracia_modelo * 100):.1f}%")

st.divider()

# LINHA 1 DE GRÁFICOS
g1, g2 = st.columns(2)

with g1:
    st.subheader("Curva de Esquecimento (Tempo de Espera)")
    df_wait = df_viz[df_viz['DiasEspera'] <= 60].groupby('DiasEspera')['Faltou'].mean().reset_index()
    df_wait['Faltou'] = df_wait['Faltou'] * 100
    fig = px.line(df_wait, x='DiasEspera', y='Faltou', 
                    labels={'DiasEspera': 'Dias de Espera (Antecedência)', 'Faltou': 'Taxa de Faltas (%)'})
    
    st.plotly_chart(fig, use_container_width=True)

with g2:
    st.subheader("Faltas por Dia")
    ordem = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado']
    
    df_dia = df_viz.groupby('DiaSemana')['Faltou'].mean().reindex(ordem).reset_index()
    df_dia['Faltou'] = df_dia['Faltou'] * 100
    
    min_y = df_dia['Faltou'].min() * 0.9
    max_y = df_dia['Faltou'].max() * 1.1
    
    fig_dia = px.bar(df_dia, x='DiaSemana', y='Faltou', color='Faltou', 
                     color_continuous_scale='Reds', range_y=[min_y, max_y],
                     labels={'DiaSemana': 'Dia da Semana', 'Faltou': 'Taxa de Faltas (%)'})
    
    st.plotly_chart(fig_dia, use_container_width=True)

# LINHA 2 DE GRÁFICOS 
st.subheader("Análise de Grupos de Risco")
cols_risco = ['Hipertensao', 'Diabetes', 'BolsaFamilia', 'Recebeu_SMS', 'Alcoolismo', 'PCD']
mapa_nomes = {
    'Hipertensao': 'Hipertensão',
    'Diabetes': 'Diabetes',
    'BolsaFamilia': 'Bolsa Família',
    'Recebeu_SMS': 'Recebeu SMS',
    'Alcoolismo': 'Alcoolismo',
    'PCD': 'PCD (Deficiência)'
}

resumo = pd.DataFrame([{'Grupo': c, 'Taxa Falta (%)': df_viz[df_viz[c]==1]['Faltou'].mean() * 100} for c in cols_risco])
resumo['Grupo'] = resumo['Grupo'].map(mapa_nomes)

fig_risco = px.bar(resumo, x='Grupo', y='Taxa Falta (%)', color='Taxa Falta (%)', 
                   color_continuous_scale='Blues', text_auto='.1f',
                   labels={'Grupo': 'Grupo de Risco', 'Taxa Falta (%)': 'Taxa de Faltas (%)'})

st.plotly_chart(fig_risco, use_container_width=True)

# --- 5. CALCULADORA ---
st.divider()
st.subheader("🤖 Simulador de Risco")
with st.form("simulador"):
    c1, c2, c3 = st.columns(3)
    with c1:
        idade = st.number_input("Idade", 0, 100, 30)
        espera = st.number_input("Dias de Espera", 0, 180, 5)
        sms = st.selectbox("Recebeu SMS?", ["Sim", "Não"])
    with c2:
        bolsa = st.checkbox("Bolsa Família")
        hiper = st.checkbox("Hipertensão")
        diab = st.checkbox("Diabetes")
    with c3:
        alco = st.checkbox("Alcoolismo")
        pcd = st.checkbox("PCD")
        submit = st.form_submit_button("Calcular")
    
    if submit:
        input_data = pd.DataFrame([[
            idade, 1 if bolsa else 0, 1 if hiper else 0, 1 if diab else 0, 
            1 if alco else 0, 1 if pcd else 0, 1 if sms == "Sim" else 0, espera
        ]], columns=feature_names)
        
        prob = model.predict_proba(input_data)[0][1] * 100
        
        st.metric("Probabilidade de Falta", f"{prob:.1f}%")
        if prob > 40: st.error("⚠️ Risco Alto")
        else: st.success("✅ Risco Baixo")

# --- RODAPÉ ---
st.divider()
st.markdown("Desenvolvido por **Larissa Vitória Gatti** | Formanda de Enfermagem & Desenvolvedora Fullstack")
st.caption("Dados utilizados: Kaggle Medical Appointment No Shows (Brasil)")
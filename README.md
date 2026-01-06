# 🏥 Análise de Faltas (No-Show) em Consultas Médicas

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Concluído-success)

## 🎯 Sobre o Projeto

Esta é uma aplicação de análise de dados desenvolvida para enfrentar um dos maiores problemas financeiros de clínicas e consultórios: o **absenteísmo (No-Show)**.

Utilizando um dataset real de agendamentos médicos, este projeto identifica padrões de comportamento e fatores de risco que levam pacientes a faltar às consultas, permitindo que gestores tomem medidas preventivas (como overbooking estratégico ou lembretes personalizados).

> **Visão de Negócio:** Uma falta não é apenas um horário vazio; é custo de oportunidade e desperdício de recursos humanos.

## 📊 Funcionalidades

- **Dashboard Interativo:** Visualização clara de indicadores de falta (KPIs).
- **Análise Temporal:** Identificação de dias da semana e horários com maior índice de cancelamento.
- **Perfil de Risco:** Correlação entre faltas e variáveis como idade, tempo de espera (lead time) e comorbidades.
- **Calculadora de Risco (Simulação):** Ferramenta para triagem rápida de probabilidade de falta baseada em dados históricos.

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python
- **Análise de Dados:** Pandas, NumPy
- **Visualização:** Plotly Express
- **Front-end / Web App:** Streamlit
- **Versionamento:** Git & GitHub

## 🚀 Como Executar

1. Clone o repositório:
   ```bash
   git clone [https://github.com/larissa-vitoria/pacientes-faltas-analise.git](https://github.com/larissa-vitoria/pacientes-faltas-analise.git)
   ```

2. Instale as dependências:
  ```bash
  pip install -r requirements.txt
  ```

3. Execute a aplicação:
  ```bash
  streamlit run app.py
  ```

## 👩‍⚕️💻 Sobre a Autora
Desenvolvido por Larissa Vitória Gatti, profissional com background híbrido único:

Tecnologia: Desenvolvedora Fullstack & Analista de Dados.

Saúde: Graduanda em Enfermagem.

Essa união permite uma visão analítica dos dados técnicos somada à compreensão real dos fluxos de atendimento em saúde.

Dados utilizados: [Medical Appointment No Shows (Kaggle)](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

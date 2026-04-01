import streamlit as st
import pandas as pd
import pickle
from google.cloud import bigquery
import unicodedata

MODEL_PATH = 'mdl-tp-categoria-conservacao-rf-top20.pkl'
TABLE_FAT = 'mackenzie-engenharia-dados.birdbase_gold.fat_aves_detalhadas'

# ================= UTILS =================

def normalize(text):
    text = str(text).lower().strip()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return text.replace(" ", "_")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def get_client():
    return bigquery.Client()

@st.cache_data
def load_gold():
    client = get_client()
    return client.query(f"SELECT * FROM `{TABLE_FAT}`").to_dataframe()

# ================= MAIN =================

def main():
    st.set_page_config(layout="centered")

    st.title("🐦 BirdBase - Classificação de Conservação")
    st.write("Preencha os dados abaixo para prever a categoria de conservação da ave.")

    model = load_model()
    features = model.feature_names_in_.tolist()
    df = load_gold()

    st.success("Modelo carregado com sucesso!")

    # ================= INPUT (IGUAL AO SEU LAYOUT) =================
    st.markdown("## 📊 Incubação e Reprodução")

    col1, col2 = st.columns(2)

    with col1:
        inc_min = st.number_input("Dias de incubação (mín)", 0, 100, 0,
                                  help="Menor tempo de incubação")
        ovos_min = st.number_input("Ovos mínimo por ninhada", 0, 20, 0)
        fled_min = st.number_input("Fledging mínimo", 0, 100, 0)

    with col2:
        inc_max = st.number_input("Dias de incubação (máx)", 0, 100, 0)
        ovos_max = st.number_input("Ovos máximo por ninhada", 0, 20, 0)
        fled_max = st.number_input("Fledging máximo", 0, 100, 0)

    st.markdown("## 🌍 Características Gerais")

    esi = st.slider("Índice ecológico", 0, 100, 5,
                    help="Quanto maior, mais especializada a espécie")

    habitats = st.slider("Qtd habitats", 0, 10, 3,
                         help="Número de ambientes onde vive")

    alimentos = st.slider("Tipos de alimentos", 0, 10, 5,
                          help="Diversidade alimentar")

    alcance = st.selectbox("Alcance restrito",
                           ["SIM", "NÃO", "NÃO INFORMADO"],
                           help="Se vive em área limitada")

    ilhas = st.selectbox("Reprodução restrita a ilhas",
                         ["SIM", "NÃO", "NÃO INFORMADO"])

    sexo = st.selectbox("Sexo depende da incubação",
                        ["SIM", "NÃO", "NÃO INFORMADO"])

    st.markdown("## 🌎 Região")
    reino = st.selectbox("Reino Biogeográfico",
                         sorted(df['nm_reino_biogeografico'].dropna().unique()))

    st.markdown("## 🍖 Dieta e Comportamento")
    dieta = st.selectbox("Dieta",
                         sorted(df['tp_dieta_portugues'].dropna().unique()))

    # ================= BOTÃO =================
    if st.button("🔎 Classificar"):

        input_data = {f: 0 for f in features}

        # numéricas
        for col in input_data:
            if 'esi' in col:
                input_data[col] = esi
            if 'habitat' in col and 'qtd' in col:
                input_data[col] = habitats
            if 'alimentos' in col:
                input_data[col] = alimentos

        # categóricas
        for f in features:
            f_norm = normalize(f)

            if normalize(reino) in f_norm:
                input_data[f] = 1
            if normalize(dieta) in f_norm:
                input_data[f] = 1
            if normalize(alcance) in f_norm:
                input_data[f] = 1
            if normalize(ilhas) in f_norm:
                input_data[f] = 1
            if normalize(sexo) in f_norm:
                input_data[f] = 1

        df_input = pd.DataFrame([input_data])[features]

        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]
        conf = round(max(proba) * 100, 2)

        # ================= ABAS =================
        tab1, tab2, tab3 = st.tabs(["🧠 Resultado", "📊 Análise", "💡 Explicação"])

        # RESULTADO
        with tab1:
            st.success(f"Categoria prevista: {pred}")
            st.metric("Confiança", f"{conf}%")

        # ANALISE (CLARA E DIRETA)
        with tab2:
            st.markdown("### 📊 Interpretação dos dados informados")

            st.write(f"• A ave possui dieta **{dieta}**, o que influencia diretamente sua forma de sobrevivência.")

            if dieta.lower() == 'carnívoro':
                st.write("→ Depende de outras espécies, podendo aumentar vulnerabilidade.")
            elif dieta.lower() == 'herbívoro':
                st.write("→ Depende da vegetação disponível.")
            elif dieta.lower() == 'invertebrados':
                st.write("→ Sensível a mudanças ambientais.")

            if habitats <= 2:
                st.write("• Vive em poucos habitats → maior risco.")
            else:
                st.write("• Vive em vários habitats → mais adaptável.")

            if esi > 50:
                st.write("• Alta especialização ecológica → mais sensível a mudanças.")
            else:
                st.write("• Baixa especialização → maior capacidade de adaptação.")

            st.write(f"• Região considerada: **{reino}**.")

        # EXPLICAÇÃO DO MODELO
        with tab3:
            st.markdown("### 💡 Como o modelo chegou nesse resultado")

            st.write("O modelo analisa padrões encontrados em dados reais de aves.")
            st.write("Ele compara as características informadas com milhares de registros.")
            st.write("A partir disso, identifica a categoria mais provável.")

            st.write("A confiança indica o nível de certeza do modelo na previsão.")

        # ================= LINK DASHBOARD =================
        st.markdown("---")
        st.markdown("### 📊 Análise completa no Dashboard")

        st.markdown(
            """
            Para uma análise mais aprofundada, acesse o dashboard interativo com dados completos:
            """
        )

        st.markdown(
            """
            <a href="https://lookerstudio.google.com/reporting/8d6f3439-8997-43c4-9a0e-164db09650fd/page/p_vuzehhyvxd" target="_blank">
                <button style="
                    background-color:#0E6FFF;
                    color:white;
                    padding:10px 20px;
                    border:none;
                    border-radius:8px;
                    cursor:pointer;
                    font-size:16px;
                ">
                    🔗 Abrir Dashboard Interativo
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()

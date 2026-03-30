import streamlit as st
import pandas as pd
import pickle
import os
import unicodedata

model_path = 'mdl-tp-categoria-conservacao-rf-top20.pkl'

# ==============================
# NORMALIZAÇÃO
# ==============================
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = text.replace(" ", "_").replace("-", "_")
    text = "_".join(filter(None, text.split("_")))
    return text

# ==============================
# OPÇÕES
# ==============================
ALCANCE_OPCOES = ["SIM", "NÃO", "NÃO INFORMADO"]
REPRO_OPCOES = ["SIM", "NÃO"]
SEXO_OPCOES = ["FÊMEA", "MACHO", "NÃO INFORMADO"]

REINO_OPCOES = [
    "L - Neotropical",
    "O - Oceania",
    "I - Indomalaio",
    "A - Afrotropical",
    "P - Paleártico"
]

DIETA_OPCOES = [
    "Frutas",
    "Invertebrados",
    "Vertebrados",
    "Néctar",
    "Sementes",
    "Onívoro"
]

# ==============================
# APP
# ==============================
def app_main():
    st.set_page_config(page_title="BirdBase", layout="centered")

    st.title("🐦 BirdBase - Classificação de Conservação")
    st.write("Preencha os dados abaixo para prever a categoria de conservação da ave.")

    if not os.path.exists(model_path):
        st.error("Modelo não encontrado.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    st.success("✅ Modelo carregado com sucesso!")

    # ==============================
    # POPUP
    # ==============================
    if "mostrar_info" not in st.session_state:
        st.session_state.mostrar_info = True

    if st.session_state.mostrar_info:
        with st.expander("ℹ️ Sobre o projeto", expanded=True):
            st.write("""
            Este sistema utiliza um modelo de Machine Learning para prever a categoria de conservação de aves.

            Você pode inserir informações como:
            - Tempo de incubação
            - Número de ovos
            - Tipo de dieta
            - Região geográfica

            ⚠️ Importante:
            Nem todas as opções disponíveis influenciam o modelo, pois ele foi treinado com um subconjunto de variáveis.

            💡 Algumas entradas podem não impactar diretamente o resultado.
            """)

            if st.button("Fechar"):
                st.session_state.mostrar_info = False

    features = model.feature_names_in_.tolist()
    input_data = {f: 0 for f in features}

    with st.form("formulario"):

        # =========================
        # INCUBAÇÃO
        # =========================
        st.markdown("## 📊 Incubação e Reprodução")

        col1, col2 = st.columns(2)

        with col1:
            dias_min = st.number_input(
                "Incubação mínima (dias)",
                min_value=0, step=1, value=0,
                help="Tempo mínimo para o ovo chocar"
            )

            ovos_min = st.number_input(
                "Ovos mínimo por ninhada",
                min_value=0, step=1, value=0,
                help="Quantidade mínima de ovos por reprodução"
            )

            fledging_min = st.number_input(
                "Fledging mínimo (dias)",
                min_value=0, step=1, value=0,
                help="Tempo mínimo para o filhote sair do ninho"
            )

        with col2:
            dias_max = st.number_input(
                "Incubação máxima (dias)",
                min_value=0, step=1, value=0,
                help="Tempo máximo para o ovo chocar"
            )

            ovos_max = st.number_input(
                "Ovos máximo por ninhada",
                min_value=0, step=1, value=0,
                help="Quantidade máxima de ovos por reprodução"
            )

            fledging_max = st.number_input(
                "Fledging máximo (dias)",
                min_value=0, step=1, value=0,
                help="Tempo máximo para o filhote sair do ninho"
            )

        # =========================
        # VALIDAÇÃO
        # =========================
        erro = False

        if dias_min > dias_max:
            st.error("Incubação mínima não pode ser maior que a máxima.")
            erro = True

        if ovos_min > ovos_max:
            st.error("Ovos mínimo não pode ser maior que o máximo.")
            erro = True

        if fledging_min > fledging_max:
            st.error("Fledging mínimo não pode ser maior que o máximo.")
            erro = True

        input_data["nr_dias_incubacao_minima"] = dias_min
        input_data["nr_dias_incubacao_maxima"] = dias_max
        input_data["nr_ovos_minimo_por_ninhada"] = ovos_min
        input_data["nr_ovos_maximo_por_ninhada"] = ovos_max
        input_data["nr_dias_periodo_fledging_minimo"] = fledging_min
        input_data["nr_dias_periodo_fledging_maximo"] = fledging_max

        # =========================
        # CARACTERÍSTICAS
        # =========================
        st.markdown("## 🌍 Características Gerais")

        indice = st.number_input(
            "Índice de especialização ecológica (ESI)",
            min_value=0, step=1, value=0,
            help="Indica o nível de dependência da espécie em relação a um habitat específico"
        )

        habitats = st.number_input(
            "Quantidade de habitats",
            min_value=0, step=1, value=0,
            help="Número de ambientes diferentes onde a espécie vive"
        )

        alimentos = st.number_input(
            "Tipos de alimentos",
            min_value=0, step=1, value=0,
            help="Diversidade alimentar da espécie"
        )

        input_data["nr_esi_indice_especializacao_ecologica"] = indice
        input_data["qtd_habitats_principais"] = habitats
        input_data["qtd_tp_alimentos_principais_consumidos"] = alimentos

        # =========================
        # CLASSIFICAÇÕES
        # =========================
        st.markdown("## 📌 Classificações")

        alcance = st.selectbox(
            "Alcance restrito",
            ALCANCE_OPCOES,
            help="Indica se a espécie vive em uma área geográfica limitada"
        )

        reproducao = st.selectbox(
            "Reprodução restrita a ilhas",
            REPRO_OPCOES,
            help="Indica se a reprodução ocorre exclusivamente em ilhas"
        )

        sexo = st.selectbox(
            "Sexo dependente da incubação",
            SEXO_OPCOES,
            help="Indica se o sexo do filhote depende da temperatura de incubação"
        )

        for opt in ALCANCE_OPCOES:
            col = f"tp_alcance_restrito_{normalize(opt)}"
            if col in input_data:
                input_data[col] = 1 if opt == alcance else 0

        for opt in REPRO_OPCOES:
            col = f"tp_reproducao_restrita_ilhas_{normalize(opt)}"
            if col in input_data:
                input_data[col] = 1 if opt == reproducao else 0

        for opt in SEXO_OPCOES:
            col = f"tp_sexo_incubacao_{normalize(opt)}"
            if col in input_data:
                input_data[col] = 1 if opt == sexo else 0

        # =========================
        # REGIÃO
        # =========================
        st.markdown("## 🌎 Região")

        reino = st.selectbox(
            "Reino biogeográfico",
            REINO_OPCOES,
            help="Região do mundo onde a espécie é encontrada"
        )

        for opt in REINO_OPCOES:
            col = f"nm_reino_biogeografico_{normalize(opt)}"
            if col in input_data:
                input_data[col] = 1 if opt == reino else 0

        # =========================
        # DIETA
        # =========================
        st.markdown("## 🥩 Dieta")

        dieta = st.selectbox(
            "Tipo de dieta",
            DIETA_OPCOES,
            help="Tipo principal de alimentação da espécie"
        )

        for opt in DIETA_OPCOES:
            col = f"tp_dieta_portugues_{normalize(opt)}"
            if col in input_data:
                input_data[col] = 1 if opt == dieta else 0

        submitted = st.form_submit_button("🔍 Classificar")

    # =========================
    # RESULTADO
    # =========================
    if submitted and not erro:
        df = pd.DataFrame([input_data])
        df = df[features]

        pred = model.predict(df)

        st.success(f"### 🧠 Categoria prevista: **{pred[0]}**")

        try:
            proba = model.predict_proba(df)
            conf = round(max(proba[0]) * 100, 2)
            st.info(f"Confiança: {conf}%")
        except:
            pass


if __name__ == "__main__":
    app_main()
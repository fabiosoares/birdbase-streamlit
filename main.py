import streamlit as st
import pandas as pd
import pickle

MODEL_PATH = 'mdl-tp-categoria-conservacao-rf-top20.pkl'

# ================= MODEL =================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# ================= SIMULATE =================

def simulate_classification(loaded_model, user_params):
    top_20_features = loaded_model.feature_names_in_.tolist()
    input_df = pd.DataFrame(columns=top_20_features)

    for feature in top_20_features:
        input_df[feature] = [0]

    for param, value in user_params.items():
        if param in top_20_features:
            input_df[param] = [value]

    prediction = loaded_model.predict(input_df)
    proba = loaded_model.predict_proba(input_df)[0]
    return prediction[0], proba

# ================= LISTAS =================

REINOS = [
    'IP - Indomalaio, Paleártico', 'A - Australiano', 'F - Afrotropical', 'L - Neotropical', 'W - Wallacea', 'M - Madagascar e Ilhas', 'AW - Australiano, Wallacea', 'E - Hemisfério Oriental', 'I - Indomalaio', 'FP - Afrotropical, Paleártico', 'O - Oceania', 'IW - Indomalaio, Wallacea', 'IO - Indomalaio, Oceania', 'PI - Paleártico, Indomalaio', 'IA - Indomalaio, Australiano', 'P - Paleártico', 'N - Neártico', 'C - Cosmopolita', 'LN - Neotropical, Neártico', 'AIW - Australiano, Indomalaio, Wallacea', 'S - Polo Sul', 'Z - Nova Zelândia e Ilhas', 'NL - Neártico, Neotropical', 'FM - Afrotropical, Madagascar e Ilhas', 'IPW - Indomalaio, Paleártico, Wallacea', 'NP - Neártico, Paleártico', 'PA - Paleártico, Australiano', 'AI - Australiano, Indomalaio', 'EZ - Hemisfério Oriental, Nova Zelândia e Ilhas', 'LS - Neotropical, Polo Sul', 'FPI - Afrotropical, Paleártico, Indomalaio', 'FI - Afrotropical, Indomalaio', 'AO - Australiano, Oceania', 'PFI - Paleártico, Afrotropical, Indomalaio', 'AOW - Australiano, Oceania, Wallacea', 'EAIFZ - Hemisfério Oriental, Australiano, Indomalaio, Afrotropical, Nova Zelândia e Ilhas', 'OZ - Oceania, Nova Zelândia e Ilhas', 'AIWZ - Australiano, Indomalaio, Wallacea, Nova Zelândia e Ilhas', 'PN - Paleártico, Neártico', 'PF - Paleártico, Afrotropical', 'AZ - Australiano, Nova Zelândia e Ilhas', 'AWZ - Australiano, Wallacea, Nova Zelândia e Ilhas', 'NOZ - Neártico, Oceania, Nova Zelândia e Ilhas', 'IFP - Indomalaio, Afrotropical, Paleártico', 'PIW - Paleártico, Indomalaio, Wallacea', 'CZ - Cosmopolita, Nova Zelândia e Ilhas', 'SLNFP - Polo Sul, Neotropical, Neártico, Afrotropical, Paleártico', 'PO - Paleártico, Oceania', 'IPA - Indomalaio, Paleártico, Australiano', 'FIPA - Afrotropical, Indomalaio, Paleártico, Australiano', 'WA - Wallacea, Australiano', 'PIAZ - Paleártico, Indomalaio, Australiano, Nova Zelândia e Ilhas', 'IAP - Indomalaio, Australiano, Paleártico', 'IPZ - Indomalaio, Paleártico, Nova Zelândia e Ilhas', 'EN - Hemisfério Oriental, Neártico', 'PIN - Paleártico, Indomalaio, Neártico', 'NPI - Neártico, Paleártico, Indomalaio', 'FMP - Afrotropical, Madagascar e Ilhas, Paleártico', 'FL - Afrotropical, Neotropical', 'AIFW - Australiano, Indomalaio, Afrotropical, Wallacea', 'IPNW - Indomalaio, Paleártico, Neártico, Wallacea', 'EAZ - Hemisfério Oriental, Australiano, Nova Zelândia e Ilhas', 'EFAIZM - Hemisfério Oriental, Afrotropical, Australiano, Indomalaio, Nova Zelândia e Ilhas, Madagascar e Ilhas', 'ASZ - Australiano, Polo Sul, Nova Zelândia e Ilhas', 'PM - Paleártico, Madagascar e Ilhas', 'EF - Hemisfério Oriental, Afrotropical', 'EAZFIM - Hemisfério Oriental, Australiano, Nova Zelândia e Ilhas, Afrotropical, Indomalaio, Madagascar e Ilhas', 'AOZ - Australiano, Oceania, Nova Zelândia e Ilhas', 'ZS - Nova Zelândia e Ilhas, Polo Sul', 'SZ - Polo Sul, Nova Zelândia e Ilhas', 'AIP - Australiano, Indomalaio, Paleártico', 'FMPIAO - Afrotropical, Madagascar e Ilhas, Paleártico, Indomalaio, Australiano, Oceania', 'AZNP - Australiano, Nova Zelândia e Ilhas, Neártico, Paleártico', 'OA - Oceania, Australiano', 'AIZ - Australiano, Indomalaio, Nova Zelândia e Ilhas', 'EIAZ - Hemisfério Oriental, Indomalaio, Australiano, Nova Zelândia e Ilhas', 'OP - Oceania, Paleártico', 'LNP - Neotropical, Neártico, Paleártico', 'CZAS - Cosmopolita, Nova Zelândia e Ilhas, Australiano, Polo Sul', 'CAZ - Cosmopolita, Australiano, Nova Zelândia e Ilhas', 'NZPFIA - Nova Zelândia e Ilhas, Paleártico, Afrotropical, Indomalaio, Australiano', 'LO - Neotropical, Oceania', 'NPLFS - Neártico, Paleártico, Neotropical, Afrotropical, Polo Sul', 'OAIFM - Oceania, Australiano, Indomalaio, Afrotropical, Madagascar e Ilhas', 'OL - Oceania, Neotropical', 'SF - Polo Sul, Afrotropical', 'OSZ - Oceania, Polo Sul, Nova Zelândia e Ilhas'
]

DIETAS = ['Frutas', 'Outras']

ALCANCE_RESTRITO  = ['SIM', 'NÃO', 'NÃO INFORMADO']
REPROD_ILHAS      = ['SIM', 'NÃO']
SEXO_INCUBACAO    = ['FEMEA', 'NÃO INFORMADO', 'OUTRO']
SEDENTARIO        = ['SIM', 'NÃO INFORMADO', 'OUTRO']

# ================= MAIN =================

def main():
    st.set_page_config(layout="centered")

    st.title("🐦 BirdBase - Classificação de Conservação")
    st.write("Preencha os dados abaixo para prever a categoria de conservação da ave.")

    model = load_model()
    st.success("Modelo carregado com sucesso!")

    # ================= INPUTS =================

    st.markdown("## 📊 Incubação e Reprodução")

    col1, col2 = st.columns(2)

    with col1:
        inc_min  = st.number_input("Dias de incubação (mín)",
                                   min_value=0, max_value=5000, value=0,
                                   help="Menor tempo de incubação registrado")
        ovos_min = st.number_input("Ovos mínimo por ninhada",
                                   min_value=0, max_value=1000, value=1)
        fled_min = st.number_input("Fledging mínimo (dias)",
                                   min_value=0, max_value=5000, value=0,
                                   help="Dias mínimos até o filhote voar")

    with col2:
        ovos_max = st.number_input("Ovos máximo por ninhada",
                                   min_value=0, max_value=1000, value=1)
        fled_max = st.number_input("Fledging máximo (dias)",
                                   min_value=0, max_value=5000, value=0,
                                   help="Dias máximos até o filhote voar")

    st.markdown("## 🌍 Características Gerais")

    esi       = st.slider("Índice ecológico (ESI)",   0, 2000, 1222,
                          help="Quanto maior, mais especializada a espécie")
    habitats  = st.slider("Qtd habitats principais",  0, 500, 2,
                          help="Número de ambientes onde a ave vive")
    alimentos = st.slider("Tipos de alimentos",       0, 500, 3,
                          help="Diversidade alimentar da espécie")

    alcance   = st.selectbox("Alcance restrito",            ALCANCE_RESTRITO)
    ilhas     = st.selectbox("Reprodução restrita a ilhas", REPROD_ILHAS)
    sexo      = st.selectbox("Sexo que incuba",             SEXO_INCUBACAO)
    sedentario = st.selectbox("Sedentário",                 SEDENTARIO)

    st.markdown("## 🌎 Região")
    reino = st.selectbox("Reino Biogeográfico", REINOS)

    st.markdown("## 🍖 Dieta")
    dieta = st.selectbox("Dieta principal", DIETAS)

    # ================= BOTÃO =================

    if st.button("🔎 Classificar"):

        user_params = {}

        # --- Numéricas ---
        user_params['nr_esi_indice_especializacao_ecologica'] = esi
        user_params['qtd_habitats_principais']                = habitats
        user_params['qtd_tp_alimentos_principais_consumidos'] = alimentos
        user_params['nr_dias_incubacao_minima']               = inc_min
        user_params['nr_ovos_minimo_por_ninhada']             = ovos_min
        user_params['nr_ovos_maximo_por_ninhada']             = ovos_max
        user_params['nr_dias_periodo_fledging_minimo']        = fled_min
        user_params['nr_dias_periodo_fledging_maximo']        = fled_max

        # --- One-hot: alcance restrito ---
        user_params['tp_alcance_restrito_SIM']          = int(alcance == 'SIM')
        user_params['tp_alcance_restrito_NÃO']          = int(alcance == 'NÃO')
        user_params['tp_alcance_restrito_NÃO INFORMADO'] = int(alcance == 'NÃO INFORMADO')

        # --- One-hot: reprodução restrita a ilhas ---
        user_params['tp_reproducao_restrita_ilhas_SIM'] = int(ilhas == 'SIM')
        user_params['tp_reproducao_restrita_ilhas_NÃO'] = int(ilhas == 'NÃO')

        # --- One-hot: sexo incubação ---
        user_params['tp_sexo_incubacao_FEMEA']          = int(sexo == 'FEMEA')
        user_params['tp_sexo_incubacao_NÃO INFORMADO']  = int(sexo == 'NÃO INFORMADO')

        # --- One-hot: sedentário ---
        user_params['tp_sedentario_SIM']                = int(sedentario == 'SIM')
        user_params['tp_sedentario_NÃO INFORMADO']      = int(sedentario == 'NÃO INFORMADO')

        # --- One-hot: reino biogeográfico ---
        user_params['nm_reino_biogeografico_L - Neotropical'] = int(reino == 'L - Neotropical')
        user_params['nm_reino_biogeografico_I - Indomalaio']  = int(reino == 'I - Indomalaio')

        # --- One-hot: dieta ---
        user_params['tp_dieta_p100ortugues_Frutas'] = int(dieta == 'Frutas')

        # --- Classificação ---
        pred, proba = simulate_classification(model, user_params)
        conf = round(max(proba) * 100, 2)

        # ================= ABAS =================
        tab1, tab2, tab3 = st.tabs(["🧠 Resultado", "📊 Análise", "💡 Explicação"])

        with tab1:
            st.success(f"Categoria prevista: **{pred}**")
            st.metric("Confiança do modelo", f"{conf}%")

            st.markdown("#### Probabilidade por categoria:")
            prob_df = pd.DataFrame({
                'Categoria': model.classes_,
                'Probabilidade (%)': [round(p * 100, 2) for p in proba]
            }).sort_values('Probabilidade (%)', ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        with tab2:
            st.markdown("### 📊 Interpretação dos dados informados")

            st.write(f"• Dieta informada: **{dieta}**")
            if dieta == 'Frutas':
                st.write("→ Depende da disponibilidade de vegetação frutífera; sensível ao desmatamento.")
            else:
                st.write("→ Dieta variada pode indicar maior adaptabilidade.")

            if habitats <= 5:
                st.write("• Vive em poucos habitats → **maior risco de extinção**.")
            elif habitats <= 15:
                st.write("• Habitats moderados → risco intermediário.")
            else:
                st.write("• Vive em muitos habitats → **maior capacidade de adaptação**.")

            if esi > 200:
                st.write("• Alta especialização ecológica (ESI elevado) → **mais sensível a perturbações**.")
            else:
                st.write("• Baixa especialização ecológica → **maior resiliência**.")

            st.write(f"• Reino Biogeográfico: **{reino}**.")
            st.write(f"• Alcance restrito: **{alcance}** | Reprodução em ilhas: **{ilhas}**.")
            st.write(f"• Sedentário: **{sedentario}**.")

        with tab3:
            st.markdown("### 💡 Como o modelo chegou nesse resultado")
            st.write("O modelo Random Forest analisa padrões em dados reais de aves.")
            st.write("Ele compara as características informadas com os registros de treinamento.")
            st.write("Cada árvore da floresta vota em uma categoria; a mais votada é a previsão final.")
            st.write("A **confiança** reflete a proporção de árvores que concordaram com a previsão.")

            st.markdown("#### Features utilizadas pelo modelo:")
            st.code("\n".join(model.feature_names_in_.tolist()))

    # ================= LINK DASHBOARD =================
    st.markdown("---")
    st.markdown("### 📊 Análise completa no Dashboard")
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
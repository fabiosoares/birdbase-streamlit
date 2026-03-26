import streamlit as st
import pandas as pd
import pickle

import re
import os


# Caminho para o seu modelo salvo no Google Drive
model_path = 'mdl-tp-categoria-conservacao-lgbm.pkl'

# Função para limpar nomes de colunas, conforme usado no treinamento
def clean_col_names(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '', str(col))
        new_cols.append(new_col)
    df.columns = new_cols
    return df

# A lógica principal do Streamlit encapsulada em uma função
def app_main():
    st.title('Previsão da Categoria de Conservação de Aves')
    st.write('Insira os parâmetros da ave abaixo para obter a previsão da sua categoria de conservação.')

    # Carregar o modelo
    if not os.path.exists(model_path):
        st.error(f"Erro: Modelo não encontrado em {model_path}. Por favor, verifique o caminho e execute as células anteriores para salvar o modelo.")
        return

    try:
        with open(model_path, 'rb') as model_pickle:
            loaded_model = pickle.load(model_pickle)
        st.success("Modelo carregado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return

    # Definição das 20 features principais e seus tipos/valores padrão
    # Isso é crucial para criar os inputs corretos no Streamlit
    features_info = {
        'nr_esi_indice_especializacao_ecologica': {'type': 'number', 'default': 120, 'min_value': 0, 'max_value': 1000, 'help': 'Índice de especialização ecológica.'},
        'nr_dias_incubacao_minima': {'type': 'number', 'default': 2000, 'min_value': 0, 'max_value': 5000, 'help': 'Número mínimo de dias de incubação.'},
        'nr_ovos_maximo_por_ninhada': {'type': 'number', 'default': 500, 'min_value': 0, 'max_value': 1000, 'help': 'Número máximo de ovos por ninhada.'},
        'nr_dias_periodo_fledging_minimo': {'type': 'number', 'default': 2000, 'min_value': 0, 'max_value': 5000, 'help': 'Número mínimo de dias no período de fledging.'},
        'qtd_habitats_principais': {'type': 'number', 'default': 300, 'min_value': 0, 'max_value': 1000, 'help': 'Quantidade de habitats principais.'},
        'qtd_tp_alimentos_principais_consumidos': {'type': 'number', 'default': 250, 'min_value': 0, 'max_value': 1000, 'help': 'Quantidade de tipos de alimentos principais consumidos.'},
        'nr_dias_periodo_fledging_maximo': {'type': 'number', 'default': 2300, 'min_value': 0, 'max_value': 5000, 'help': 'Número máximo de dias no período de fledging.'},
        'ds_ordem_Passarinhosemgeralpardaiscanriossabisetc': {'type': 'boolean', 'default': False, 'help': 'Pertence à ordem de Passarinhos em geral (pardais, canários, etc.).'},
        'tp_alcance_restrito_NOINFORMADO': {'type': 'boolean', 'default': False, 'help': 'Informação sobre alcance restrito não disponível.'},
        'tp_alcance_restrito_NO': {'type': 'boolean', 'default': True, 'help': 'Não possui alcance restrito.'},
        'nr_ovos_minimo_por_ninhada': {'type': 'number', 'default': 300, 'min_value': 0, 'max_value': 1000, 'help': 'Número mínimo de ovos por ninhada.'},
        'tp_reproducao_restrita_ilhas_NO': {'type': 'boolean', 'default': True, 'help': 'A reprodução não é restrita a ilhas.'},
        'nr_dias_incubacao_maxima': {'type': 'number', 'default': 2100, 'min_value': 0, 'max_value': 5000, 'help': 'Número máximo de dias de incubação.'},
        'nm_reino_biogeografico_LNeotropical': {'type': 'boolean', 'default': False, 'help': 'Reino biogeográfico Neotropical.'},
        'nm_reino_biogeografico_AAustraliano': {'type': 'boolean', 'default': False, 'help': 'Reino biogeográfico Australiano.'},
        'tp_sexo_incubacao_NOINFORMADO': {'type': 'boolean', 'default': True, 'help': 'Informação sobre sexo de incubação não disponível.'},
        'nm_reino_biogeografico_IIndomalaio': {'type': 'boolean', 'default': False, 'help': 'Reino biogeográfico Indomalaio.'},
        'tp_comportamento_criacao_cooperativa_NO': {'type': 'boolean', 'default': True, 'help': 'Não apresenta comportamento de criação cooperativa.'},
        'tp_dieta_portugues_Invertebrados': {'type': 'boolean', 'default': False, 'help': 'Dieta principal é de Invertebrados.'},
        'ds_habitat_principal_Florestaincluiflorestasecundriapntanotaigaetc': {'type': 'boolean', 'default': True, 'help': 'Habitat principal inclui floresta (secundária, pântano, taiga, etc.).'}
    }

    input_data = {}
    st.subheader("Parâmetros da Ave:")
    for feature, info in features_info.items():
        display_name = feature.replace('_', ' ').title()
        # Usando st.expander para organizar melhor os inputs
        with st.expander(f"**{display_name}**", expanded=False):
            if info['type'] == 'number':
                input_data[feature] = st.number_input(
                    f"Valor para {display_name}:",
                    min_value=info.get('min_value'),
                    max_value=info.get('max_value'),
                    value=info.get('default'),
                    help=info.get('help'),
                    key=feature # Chave única para o widget
                )
            elif info['type'] == 'boolean':
                # Convertendo True/False para 1/0 para features one-hot encoded
                input_data[feature] = int(st.checkbox(
                    f"Marque se a característica '{display_name}' se aplica (1=Sim):",
                    value=info.get('default'),
                    help=info.get('help'),
                    key=feature
                ))

    if st.button('Prever Categoria de Conservação', help='Clique para fazer a previsão com os parâmetros inseridos.'):
        # Criar DataFrame com os novos dados
        new_data_for_prediction = pd.DataFrame([input_data])

        # Limpar os nomes das colunas para corresponder ao treinamento do modelo
        new_data_for_prediction = clean_col_names(new_data_for_prediction.copy())

        # Fazer a predição
        try:
            sample_prediction = loaded_model.predict(new_data_for_prediction)
            st.success(f"### A categoria de conservação prevista é: **{sample_prediction[0]}**")
        except Exception as e:
            st.error(f"Erro ao fazer a previsão: {e}\nVerifique se todos os parâmetros foram inseridos corretamente.")

# Executar o aplicativo Streamlit diretamente no Colab
# st.run_headless() permite que o Streamlit seja renderizado no output da célula
st.run_headless(app_main)

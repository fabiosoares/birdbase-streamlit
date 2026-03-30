# 🐦 BirdBase - Classificação de Conservação

Aplicação interativa desenvolvida em **Streamlit** para previsão da categoria de conservação de aves utilizando **Machine Learning**.

---

## 🚀 Sobre o Projeto

O BirdBase permite que o usuário insira características biológicas e ecológicas de uma ave para prever sua categoria de conservação.

A aplicação combina:

* Interface interativa (Streamlit)
* Pipeline de dados (ETL)
* Modelo de Machine Learning (Random Forest)

---

## 🧠 Como funciona

O sistema recebe informações como:

* Tempo de incubação
* Número de ovos por ninhada
* Período de desenvolvimento (fledging)
* Tipo de dieta
* Região biogeográfica
* Características ecológicas

Esses dados são processados e enviados para um modelo treinado que retorna a **categoria de conservação prevista**.

---

## 🔄 Arquitetura do Projeto

A aplicação foi construída com separação entre:

### 🔹 Interface (UI)

Responsável pela entrada de dados do usuário, com uma experiência amigável e acessível.

### 🔹 Modelo de Machine Learning

Responsável pela predição, utilizando um conjunto reduzido de variáveis selecionadas durante o treinamento.

---

## ⚠️ Importante sobre as Features

Nem todas as opções disponíveis na interface são utilizadas pelo modelo.

Isso acontece porque:

* O modelo foi treinado com um subconjunto das variáveis (features)
* A interface representa uma visão mais completa da base de dados

👉 Portanto:

* Algumas entradas podem não impactar diretamente a previsão
* O sistema ignora automaticamente variáveis que não fazem parte do modelo

---

## 💡 Diferenciais do Projeto

* Interface amigável para usuários não técnicos
* Validação de dados (ex: valores mínimos e máximos)
* Explicação contextual dos campos (tooltips)
* Separação entre dados, modelo e apresentação
* Pronto para deploy em produção

---

## 🛠️ Tecnologias Utilizadas

* Python
* Streamlit
* Pandas
* Scikit-learn

---

## 🔗 Notebooks do Projeto

* Modelo Random Forest:
  https://colab.research.google.com/drive/1D0VcOnqtN_BBFxeDq1EJyYNiQyKZo8s3

* Desenvolvimento inicial:
  https://colab.research.google.com/drive/1adWwofO6oPmwkb82c0esgW-gmV0KUuTF

---

## 📊 Objetivo

Demonstrar na prática a aplicação de:

* Engenharia de Dados
* Machine Learning
* Desenvolvimento de aplicações interativas

em um cenário real de análise ecológica.

---

## 🚀 Próximos Passos

* Explicabilidade do modelo (feature importance)
* Melhorias de UI/UX
* Deploy em ambiente público
* Integração com API

---

## 👨‍💻 Autor

Projeto desenvolvido para fins acadêmicos e evolução profissional.

# 1-Classificação de Discurso de Ódio

Projeto voltado para a construção e avaliação de modelos de classificação de discurso de ódio em textos em português.

O objetivo é identificar se uma publicação contém linguagem ofensiva ou discurso de ódio, utilizando técnicas de Processamento de Linguagem Natural (NLP) e modelos de Machine Learning.

A base utilizada é a **OLID-BR**, composta por textos rotulados para identificação de conteúdo ofensivo. O projeto também aborda desafios comuns desse tipo de problema, como desbalanceamento das classes e avaliação adequada dos modelos.

## Principais tópicos abordados

- Carregamento e preparação dos dados  
- Análise exploratória das classes  
- Limpeza e pré-processamento dos textos  
- Vetorização utilizando técnicas de NLP  
- Treinamento de modelos de classificação  
- Comparação de desempenho entre modelos  
- Tratamento de dados desbalanceados  
- Avaliação por métricas como acurácia, precision, recall e F1-score  
- Análise de matriz de confusão  
- Discussão sobre limitações e possíveis aplicações práticas  

## Tecnologias utilizadas

- Python  
- Pandas  
- Scikit-learn  
- NLTK  
- Matplotlib  
- Seaborn  

## Notebook

[Abrir no Google Colab](https://colab.research.google.com/)


# 2- Análise de Sentimentos de Avaliações da Amazon

Projeto voltado para a construção e avaliação de modelos de análise de sentimentos em avaliações de produtos alimentícios da Amazon.

O objetivo é classificar reviews como positivas ou negativas a partir do texto das avaliações, utilizando técnicas de Processamento de Linguagem Natural (NLP) e modelos de Machine Learning.

A base utilizada é a **Amazon Fine Food Reviews**, que reúne mais de 500 mil avaliações de produtos alimentícios publicados na Amazon entre 1999 e 2012. Para transformar o problema em classificação binária, avaliações com nota maior que 3 foram consideradas positivas, enquanto notas menores ou iguais a 3 foram classificadas como negativas.

## Principais tópicos abordados

- Carregamento e preparação dos dados  
- Remoção de avaliações duplicadas  
- Construção da variável-alvo a partir da nota da avaliação  
- Análise do desbalanceamento das classes  
- Amostragem dos dados para otimizar o processamento  
- Limpeza e pré-processamento dos textos  
- Remoção de tags HTML, pontuações e stopwords  
- Lematização com spaCy  
- Vetorização utilizando Bag of Words e TF-IDF  
- Treinamento de modelos de Regressão Logística, Naive Bayes e SVM  
- Comparação de desempenho entre modelos  
- Avaliação por acurácia, F1-score e matriz de confusão  
- Teste do modelo com novas avaliações  

## Tecnologias utilizadas

- Python  
- Pandas  
- NLTK  
- spaCy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- KaggleHub  

## Dataset

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## Notebook

[Abrir no Google Colab](https://colab.research.google.com/)

3- # Análise de Letras do One Direction com TF-IDF

Projeto de Processamento de Linguagem Natural (NLP) voltado para a coleta, limpeza e análise das letras das músicas da banda One Direction.

O objetivo é identificar as palavras mais relevantes presentes nas músicas do grupo utilizando a técnica de TF-IDF (*Term Frequency–Inverse Document Frequency*). A análise permite observar quais termos possuem maior importância no conjunto das letras e em músicas específicas.

As letras são coletadas automaticamente a partir do site Vagalume e passam por etapas de pré-processamento antes da geração da matriz TF-IDF.

## Principais tópicos abordados

- Web scraping das músicas e letras do One Direction  
- Coleta de informações como nome da música, álbum e letra  
- Limpeza e pré-processamento dos textos em inglês  
- Conversão dos textos para letras minúsculas  
- Remoção de stopwords e ruídos comuns em músicas  
- Lematização utilizando spaCy  
- Tokenização das letras  
- Vetorização utilizando TF-IDF  
- Criação de uma matriz com a importância das palavras por música  
- Identificação das palavras mais relevantes no conjunto de letras  
- Análise das palavras mais importantes de uma música específica  

## Tecnologias utilizadas

- Python  
- Requests  
- BeautifulSoup  
- Pandas  
- spaCy  
- Scikit-learn  

## Fonte dos dados

As letras das músicas são coletadas do site Vagalume.

## Notebook

[Abrir no Google Colab](https://colab.research.google.com/)

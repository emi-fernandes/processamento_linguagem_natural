# Classificação de Discurso de Ódio com Ambiguidade — OLID-BR

Projeto de Processamento de Linguagem Natural (PLN) para classificar comentários em português como **ofensivos** ou **não ofensivos** e, em uma segunda etapa, separar **ofensas comuns** de **discurso de ódio direcionado a grupos protegidos**.

O trabalho usa o dataset **OLID-BR** e enfatiza casos ambíguos, como crítica política, ironia, negação, gírias, eufemismos e *dog-whistles*.

## Objetivos

- Detectar comentários ofensivos (`OFF`) e não ofensivos (`NOT`).
- Reduzir os efeitos do desbalanceamento entre as classes.
- Identificar, entre comentários ofensivos, aqueles que configuram discurso de ódio direcionado.
- Examinar erros e limites de modelos baseados em palavras, especialmente em textos ambíguos.

## Abordagem

O pipeline é composto por duas etapas:

1. **Classificação de ofensa:** distingue `OFF` de `NOT`.
2. **Classificação de discurso de ódio:** aplicada apenas aos textos classificados como ofensivos, distinguindo `Ofensa Comum` de `Discurso de Ódio`.

O rótulo `hate_speech` é construído a partir de comentários direcionados (`TIN`) associados a categorias como racismo, sexismo, LGBTQIA+fobia, xenofobia, intolerância religiosa, ideologia, saúde e estilo de vida.

## Técnicas utilizadas

- Limpeza de texto: remoção de URLs, menções, hashtags, símbolos e números.
- Tokenização, remoção de *stopwords* e lematização com spaCy (`pt_core_news_sm`).
- Preservação de negações como “não”, “nem”, “nunca” e “jamais”.
- Vetorização com **Bag of Words** e **TF-IDF**, usando unigramas e bigramas.
- Classificadores de **Regressão Logística**.
- Estratégias para desbalanceamento:
  - `class_weight='balanced'`
  - ajuste de limiar de decisão (*threshold tuning*)
  - SMOTE aplicado somente aos dados de treino

## Resultados principais

O baseline sem balanceamento tende a prever a classe majoritária (`OFF`), alcançando cerca de 85% de acurácia, mas com **F1 para a classe `NOT` igual a 0**.

As estratégias balanceadas melhoram a cobertura da classe minoritária. No notebook, a melhor métrica para `NOT` é obtida com **Bag of Words + Regressão Logística + `class_weight='balanced'`**, com F1-NOT aproximado de **0,43**. Modelos com TF-IDF, ajuste de limiar e SMOTE apresentam resultados próximos, com diferentes compromissos entre F1, acurácia e sensibilidade.

## Limitações

Modelos BoW e TF-IDF não compreendem contexto semântico, ordem das palavras ou pragmática. Por isso, ainda apresentam dificuldades em:

- distinguir crítica política de ataques a pessoas ou grupos;
- interpretar negação e ironia;
- detectar linguagem indireta e *dog-whistles*;
- acompanhar gírias, neologismos e termos codificados novos;
- separar com precisão insulto genérico de ódio dirigido a grupos protegidos.

## Próximos passos

- Usar modelos contextuais em português, como BERTimbau.
- Incorporar léxicos de linguagem codificada e *dog-whistles*.
- Adotar classificação com abstenção para previsões de baixa confiança.
- Melhorar e revisar a anotação de casos ambíguos no conjunto de dados.

## Como executar

1. Abra o notebook `classificacao_discurso_odio-2.ipynb` em Jupyter Notebook, JupyterLab ou Google Colab.
2. Execute as células de instalação:

```bash
pip install datasets spacy imbalanced-learn
python -m spacy download pt_core_news_sm
```

3. Execute o notebook na ordem apresentada. Os dados são carregados diretamente do repositório do dataset OLID-BR no Hugging Face.

## Dependências

- Python 3
- pandas
- numpy
- matplotlib
- nltk
- spaCy
- scikit-learn
- imbalanced-learn
- datasets

## Autores

**Grupo D** — Daniel Mafezolli, Emilly Fernandes, Ewerton Arrais, João Gabriel Tasca e Miguel Veiga.

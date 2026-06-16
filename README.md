Vou explicar cada seção em detalhes, na ordem do notebook.

---

## Seção 1 — Instalação e Imports

```python
!pip install datasets spacy imbalanced-learn --quiet
!python -m spacy download pt_core_news_sm --quiet
```
Instala 3 bibliotecas extras que não vêm por padrão: `datasets` (para baixar o OLID-BR do HuggingFace), `spacy` (para lematização em português) e `imbalanced-learn` (para o SMOTE). O modelo `pt_core_news_sm` é o dicionário do spaCy para português.

```python
import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ...
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
```
Cada import tem um papel: `pandas` manipula tabelas, `re` faz buscas com expressões regulares, `numpy` faz cálculos numéricos, `matplotlib` gera gráficos, `sklearn` fornece os modelos de ML.

---

## Seção 2 — Carregamento do Dataset

```python
splits = {
    'train': 'data/train-00000-of-00001-0d8933b80051ca0e.parquet',
    'test':  'data/test-00000-of-00001-914dbee7561d2266.parquet'
}
df_train = pd.read_parquet('hf://datasets/dougtrajano/olid-br/' + splits['train'])
df_test  = pd.read_parquet('hf://datasets/dougtrajano/olid-br/' + splits['test'])
```
O formato `.parquet` é como um CSV otimizado — mais rápido e mais leve. O prefixo `hf://` é um atalho da biblioteca `datasets` para baixar direto do HuggingFace sem precisar fazer download manual.

---

## Seção 2.1 — Criação do Label hate_speech

Esta é uma das decisões mais importantes do trabalho:

```python
colunas_hate = ['health', 'ideology', 'lgbtqphobia', 'other_lifestyle',
                'racism', 'sexism', 'xenophobia']

def cria_label_hate(df):
    temp = df[colunas_hate].fillna(0)
    df['hate_speech'] = (
        (df['is_targeted'] == 'TIN')
        &
        (temp.sum(axis=1) > 0)
    ).astype(int)
    return df
```

O dataset original só tem `is_offensive` (OFF/NOT). O grupo criou um novo rótulo combinando **duas condições com `&` (E lógico)**:

- `is_targeted == 'TIN'` — o ataque é direcionado a um alvo específico
- `temp.sum(axis=1) > 0` — pelo menos uma coluna de grupo protegido é verdadeira

O `fillna(0)` substitui valores vazios por zero para não quebrar o `.sum()`. O `.astype(int)` converte `True/False` em `1/0`. Se as **duas condições forem verdadeiras ao mesmo tempo**, é discurso de ódio — caso contrário, é só ofensa comum.

---

## Seção 3 — Análise do Desbalanceamento

```python
counts = df['is_offensive'].value_counts()
pct    = df['is_offensive'].value_counts(normalize=True) * 100
```

`value_counts()` conta quantas vezes cada valor aparece. `normalize=True` transforma em proporção (0 a 1), e multiplicar por 100 vira percentual. O gráfico gerado mostra visualmente a razão ~5.8:1 entre OFF e NOT.

```python
ratio = (df_train['is_offensive'].value_counts()['OFF'] /
         df_train['is_offensive'].value_counts()['NOT'])
```
Divide o total de OFF pelo total de NOT para calcular a razão exata de desbalanceamento.

---

## Seção 4 — Exploração de Grupos Protegidos

```python
df_off = df_train[df_train['is_offensive'] == 'OFF'].copy()
```
Filtra só os comentários ofensivos. O `.copy()` é importante — sem ele, alterações nesse DataFrame afetariam o original (comportamento do pandas chamado de "SettingWithCopyWarning").

```python
pct_direcionado = (df_off['is_targeted'] == 'TIN').sum() / len(df_off) * 100
```
Conta quantos registros têm `is_targeted == 'TIN'`, divide pelo total de ofensivos e multiplica por 100. Resultado: a porcentagem de ofensas que são direcionadas.

O loop de gráficos depois percorre cada coluna de grupo protegido (`racism`, `sexism`, etc.) e plota a distribuição — mostrando quais tipos de ódio aparecem mais no dataset.

---

## Seção 5 — Pré-processamento

```python
spc_pt = spacy.load('pt_core_news_sm')
stopwords_pt = stopwords.words('portuguese')

for palavra in ['nao', 'nem', 'nunca', 'jamais']:
    if palavra in stopwords_pt:
        stopwords_pt.remove(palavra)
```
Carrega o modelo de português do spaCy. As stopwords são palavras muito comuns que não carregam significado (artigos, preposições). O loop **remove as negações da lista de stopwords** — isso é inteligente porque "não é racismo" e "é racismo" têm sentidos opostos, e sem "não" o modelo perderia essa distinção.

```python
def limpa_texto(texto):
    texto = re.sub(r'http\S+|www\S+', '', texto)   # remove URLs
    texto = re.sub(r'@\w+|#\w+', '', texto)         # remove @menções e #hashtags
    texto = texto.lower()                            # minúsculas
    texto = re.sub(r'[\W\d_]+', ' ', texto)         # remove pontuação e números
    tokens = [p for p in texto.split() if p not in stopwords_pt]  # remove stopwords
    doc = spc_pt(' '.join(tokens))
    return ' '.join([t.lemma_ for t in doc])        # lematiza
```

Cada `re.sub(padrão, substituto, texto)` é uma busca e substituição com regex. O `\S+` significa "qualquer caractere não-espaço". O `\W` significa "qualquer caractere que não seja letra ou número". A lematização com `t.lemma_` reduz cada palavra à sua forma base — "correndo", "correu", "correr" viram todas "correr".

```python
df_train['text_clean'] = df_train['text'].apply(limpa_texto)
```
O `.apply()` aplica a função em cada linha da coluna — é o equivalente a um `for` mas muito mais eficiente.

---

## Seção 6 — Vetorização

Modelos de ML não entendem texto — só números. Os vetorizadores convertem texto em matrizes numéricas.

```python
vectorizer_bow = CountVectorizer(max_features=5000, ngram_range=(1,2))
vectorizer_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_bow   = vectorizer_bow.fit_transform(X_train_text)
X_test_bow    = vectorizer_bow.transform(X_test_text)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train_text)
X_test_tfidf  = vectorizer_tfidf.transform(X_test_text)
```

`max_features=5000` mantém só as 5000 palavras/bigramas mais frequentes. `ngram_range=(1,2)` captura unigramas ("ódio") e bigramas ("discurso ódio") — bigramas capturam expressões que palavras isoladas não capturam.

**Diferença crucial entre os dois:**
- **BoW (`CountVectorizer`)**: cada célula da matriz é a **contagem** de vezes que a palavra aparece no texto
- **TF-IDF (`TfidfVectorizer`)**: cada célula é um **peso** que aumenta se a palavra é rara no corpus todo — palavras muito comuns como "que" valem quase zero, palavras raras e específicas valem mais

O `.fit_transform()` no treino aprende o vocabulário E transforma. O `.transform()` no teste só transforma — **sem reaprender** — porque o modelo nunca pode "ver" o teste durante o treino.

---

## Seções 8 a 11 — As 4 Estratégias

### Seção 8 — Baseline
```python
rl_base = LogisticRegression(max_iter=1000, random_state=42)
rl_base.fit(X_train_bow, y_train)
y_pred_base = rl_base.predict(X_test_bow)
```
Regressão Logística simples. `max_iter=1000` aumenta o limite de iterações para garantir convergência. `random_state=42` garante reprodutibilidade. Resultado: F1-NOT = 0.00 — classifica tudo como OFF.

### Seção 9 — Class Weight
```python
rl_bow_cw = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
```
O parâmetro `class_weight='balanced'` faz o sklearn calcular automaticamente pesos inversamente proporcionais à frequência de cada classe. Se OFF aparece 6x mais que NOT, o erro em NOT é penalizado 6x mais durante o treino — forçando o modelo a prestar atenção na classe minoritária.

### Seção 10 — Threshold Tuning
```python
thresholds = np.arange(0.1, 0.9, 0.01)
y_proba = rl_tfidf_cw.predict_proba(X_test_tfidf)[:, 1]

best_thr, best_f1 = 0.5, 0
for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    f1 = f1_score(y_test, y_pred_thr, average='macro')
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
```
`predict_proba()` retorna a probabilidade de cada classe — o `[:, 1]` pega só a coluna da classe positiva (OFF). Em vez de usar o threshold padrão de 0.5, o loop testa todos os valores de 0.10 a 0.89 em passos de 0.01 e guarda o que dá melhor F1-macro. O melhor encontrado foi 0.41 — abaixo de 0.5 porque o modelo está "viesado" para prever NOT mesmo com balanceamento.

### Seção 11 — SMOTE
```python
smote = SMOTE(random_state=42)
X_train_tfidf_sm, y_train_sm = smote.fit_resample(X_train_tfidf, y_train)
```
SMOTE (Synthetic Minority Oversampling Technique) cria exemplos **artificiais** da classe minoritária. Ele não duplica exemplos existentes — pega dois exemplos de NOT próximos no espaço vetorial e gera um ponto intermediário entre eles. O resultado foi 762 → 4452 exemplos de NOT. **Aplicado só no treino** — nunca no teste, porque o teste deve refletir o mundo real.

---

## Seção 12 — Comparação Final

```python
resultados = pd.DataFrame([r_base, r_bow_cw, r_tfidf_cw, r_threshold, r_smote])
print(resultados[['titulo', 'acc', 'f1_macro', 'f1_not', 'f1_off']].to_string(index=False))
```
Cada `r_*` é um dicionário com as métricas de cada modelo. O `pd.DataFrame()` empilha todos numa tabela para comparação direta. O gráfico de barras agrupadas gerado depois mostra visualmente que o baseline tem a maior acurácia mas o pior F1-macro — provando que acurácia sozinha é uma métrica enganosa em bases desbalanceadas.

---

## Seção 13 — Palavras Mais Relevantes

```python
feature_names = vectorizer_tfidf.get_feature_names_out()
coefs = rl_tfidf_cw.coef_[0]
top_off = pd.Series(coefs, index=feature_names).nlargest(15)
top_not = pd.Series(coefs, index=feature_names).nsmallest(15)
```
A Regressão Logística aprende um **coeficiente** para cada palavra. Coeficientes positivos altos indicam palavras que empurram para OFF; coeficientes negativos baixos indicam palavras que empurram para NOT. O `.nlargest(15)` e `.nsmallest(15)` pegam os extremos — as palavras mais "decisivas" para cada classe.

---

## Seção 14 — Análise de Erros

```python
df_analise = df_test.copy().reset_index(drop=True)
df_analise['y_true'] = y_test.values
df_analise['y_pred'] = y_pred_tfidf_cw
df_analise['proba_off'] = rl_tfidf_cw.predict_proba(X_test_tfidf)[:, 1]

fp = df_analise[(df_analise['y_true'] == 0) & (df_analise['y_pred'] == 1)]
fn = df_analise[(df_analise['y_true'] == 1) & (df_analise['y_pred'] == 0)]
```
Adiciona as predições e probabilidades ao DataFrame de teste para poder filtrar os erros. `fp` são os Falsos Positivos (real=NOT, previsto=OFF) e `fn` são os Falsos Negativos (real=OFF, previsto=NOT). O filtro duplo com `&` garante que ambas as condições sejam verdadeiras ao mesmo tempo.

**Dog-whistles:**
```python
dog_whistles = ['lugar de mulher', 'coisa de viado', 'esses nordestinos', ...]
for dw in dog_whistles:
    ocorrencias = fn[fn['text'].str.lower().str.contains(dw, na=False)]
```
Busca nos falsos negativos textos que contenham esses padrões. O `na=False` evita erro quando o texto é `NaN`. Se encontrar, mostra os exemplos — comprovando que o modelo falha nesses casos.

**Zona de incerteza:**
```python
zona_incerta = df_analise[
    (df_analise['proba_off'] >= 0.35) & (df_analise['proba_off'] <= 0.65)
]
```
Filtra comentários onde o modelo está em dúvida — probabilidade entre 35% e 65%. Esses 743 casos são exatamente a fronteira entre crítica, ofensa e ódio que o trabalho propõe investigar.

---

## Seção 15 — Pipeline Final

```python
def classifica_comentario(texto):
    texto_limpo = limpa_texto(texto)
    vec = vectorizer_tfidf.transform([texto_limpo])

    # Etapa 1: ofensivo ou não?
    pred_off = rl_tfidf_cw.predict(vec)[0]
    if pred_off == 0:
        return "NÃO OFENSIVO"

    # Etapa 2: ódio ou ofensa comum?
    pred_hate = hate_model.predict(vec)[0]
    if pred_hate == 1:
        return "DISCURSO DE ÓDIO"

    return "OFENSA COMUM"
```

O fluxo é sequencial e com **early return** — se já decidiu que não é ofensivo na Etapa 1, nem entra na Etapa 2. O `[0]` no final do `.predict()` é porque o método sempre retorna um array, mesmo para um único texto — o `[0]` pega o primeiro (e único) elemento. O `hate_model` é um segundo classificador treinado **só sobre os comentários ofensivos**, distinguindo ofensa comum de discurso de ódio.

---

Em resumo, o notebook segue uma lógica muito clara: carrega e entende os dados → limpa e transforma o texto → testa múltiplas estratégias para lidar com o desbalanceamento → analisa onde e por que o modelo erra → implementa um pipeline hierárquico de 2 etapas que gera 3 classes de saída. Quer que eu aprofunde algum conceito específico?

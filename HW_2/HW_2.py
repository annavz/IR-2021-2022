import zipfile
import os
import pandas as pd
import numpy as np
import pymorphy2
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#  подготовка файлов
path_to_file = input('Введите путь до файла: ')
path = r'./'
if not os.path.exists(path):
    os.mkdir(path)
with zipfile.ZipFile(path_to_file, 'r') as zip_ref:
    zip_ref.extractall(path)

#  инициализация
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")
russian_stopwords = [morph.parse(word)[0].normal_form for word in russian_stopwords]
tfidf_vectorizer = TfidfVectorizer()


def get_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return pd.DataFrame(data=all_files, columns=['filename'])


def preprocessing(text, morph=morph, russian_stopwords=russian_stopwords):
    text = re.sub('\n', ' ', text, flags=re.DOTALL)
    preprocessed_text = []
    text = nltk.word_tokenize(text)
    for token in text:
        if re.sub('-', '', token).isalpha():
            token = morph.parse(token)[0].normal_form.lower()
            if token not in russian_stopwords:
                preprocessed_text.append(token)
    return ' '.join(preprocessed_text)


def create_inverse_index(corpus, vectorizer):
    X = vectorizer.fit_transform(corpus)
    inverse_index = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())
    return vectorizer, inverse_index


def process_query(text, vectorizer):
    return vectorizer.transform([preprocessing(text)]).toarray()


def calculate_cosine_similarity(query_vec, docs):
    df = pd.DataFrame([])
    df['filename'] = docs[docs.columns[0]].transform(lambda x: x.split('\\')[-1].strip(' '))
    df['similarity'] = cosine_similarity(docs[docs.columns[1:]], query_vec)
    return df.sort_values('similarity', ascending=False)


def search(path, vectorizer):
    df = get_files(os.path.join(path, 'friends-data'))
    df['text'] = np.nan
    for num, row in df.iterrows():
        with open(row['filename'], 'r', encoding='utf-8-sig') as f:
            text = f.read()
        df.loc[num, 'text'] = text
    #  подготовка корпуса
    print('Происходит обработка корпуса')
    df['preprocessed_text'] = df['text'].transform(lambda x: preprocessing(x))
    vectorizer, inverse_index = create_inverse_index(df['preprocessed_text'], vectorizer)
    df1 = pd.concat([df[['filename']], inverse_index], axis=1)
    #  подготовка запроса
    query = input('Введите запрос: ')
    query = process_query(query, vectorizer)
    df2 = calculate_cosine_similarity(query, df1)
    return df2


print(search(path, tfidf_vectorizer))

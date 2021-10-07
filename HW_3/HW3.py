import pandas as pd
import numpy as np
import pymorphy2
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from scipy import sparse
import os
import json


def get_max_value(answer_list):
    if answer_list:
        for num, answer in enumerate(answer_list):
            if answer['author_rating']['value'] != '':
                answer_list[num]['author_rating']['value'] = int(answer['author_rating']['value'])
            else:
                answer_list[num]['author_rating']['value'] = 0
        return max(answer_list, key=lambda x: x['author_rating']['value'])['text']
    else:
        return np.nan


def preprocessing(text, morph, russian_stopwords):
    text = re.sub('\n', ' ', text, flags=re.DOTALL)
    preprocessed_text = []
    text = nltk.word_tokenize(text)
    for token in text:
        if re.sub('-', '', token).isalpha():
            token = morph.parse(token)[0].normal_form.lower()
            if token not in russian_stopwords:
                preprocessed_text.append(token)
    return ' '.join(preprocessed_text)


def get_corpus(morph, russian_stopwords):
    path = input('Введите путь до файла: ')
    path = os.path.join(path, 'questions_about_love.jsonl')
    with open(path, 'r', encoding='utf-8') as f:
        corpus = '[' + ', '.join(f.readlines()) + ']'
    corpus = json.loads(corpus)
    corpus = [question['answers'] for question in corpus]
    corpus = [get_max_value(answer_list) for answer_list in corpus]
    corpus = [answer for answer in corpus if pd.isnull(answer) == False][:50000]
    print('Тексты проходят предобработку: ')
    preprocessed_corpus = [preprocessing(c, morph, russian_stopwords) for c in tqdm(corpus)]
    return corpus, preprocessed_corpus


def get_inverse_index(corpus):
    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    x_count_vec = count_vectorizer.fit_transform(corpus)
    x_tf_vec = tf_vectorizer.fit_transform(corpus)
    tfidf_vectorizer.fit(corpus)
    idf = tfidf_vectorizer.idf_
    k = 2
    b = 0.75
    len_d = np.squeeze(np.array(x_count_vec.sum(axis=1)))
    average_len_d = x_count_vec.sum() / x_count_vec.shape[0]

    rows = []
    columns = []
    values = []
    for i, j in zip(*x_tf_vec.nonzero()):
        num = idf[j] * x_tf_vec[i, j] * (k + 1)
        den = (k * (1 - b + b * len_d[i] / average_len_d)) + x_tf_vec[i, j]
        value = num / den
        rows.append(i)
        columns.append(j)
        values.append(value)
    return sparse.csr_matrix((values, (rows, columns))), count_vectorizer


def calculate_closeness(inverse_index, query):
    rating = inverse_index.dot(query)
    rating = np.squeeze(rating.toarray())
    rating_args = rating.argsort()[::-1]
    return rating, rating_args


def search():
    #  инициализация
    morph = pymorphy2.MorphAnalyzer()
    russian_stopwords = stopwords.words("russian")
    russian_stopwords = [morph.parse(word)[0].normal_form for word in russian_stopwords]

    corpus, preprocessed_corpus = get_corpus(morph, russian_stopwords)
    inverse_index, count_vectorizer = get_inverse_index(preprocessed_corpus)
    # обработка запроса
    query = input('Введите запрос: ')
    query = preprocessing(query, morph, russian_stopwords)
    query = count_vectorizer.transform([query]).T
    # подсчет близости
    rating, rating_args = calculate_closeness(inverse_index, query)
    # вывод
    df_rating = pd.DataFrame([])
    df_rating['answer'] = np.array(corpus)[rating_args]
    df_rating['BM25'] = rating[rating_args]
    print(df_rating)


if __name__ == '__main__':
    search()
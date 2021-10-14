import pandas as pd
import numpy as np
import pymorphy2
import nltk
from nltk.corpus import stopwords
import re
from tqdm import tqdm
import os
import json
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse


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


def get_corpus():
    path = input('Введите путь до файла: ')
    path = os.path.join(path, 'questions_about_love.jsonl')
    with open(path, 'r', encoding='utf-8') as f:
        corpus = '[' + ', '.join(f.readlines()) + ']'
    corpus = json.loads(corpus)
    questions = [question['question'] for question in corpus if question['answers'] != []][:10000]
    answers = [question['answers'] for question in corpus if question['answers'] != []][:10000]
    answers = [get_max_value(answer_list) for answer_list in answers]
    return questions, answers


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


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


def transform_text(texts, vectorization_method, model):
    print('Тексты векторизируются: ')
    if vectorization_method == 'fasttext':
        corpus = []
        for text in tqdm(texts):
            text = text.split(' ')
            text_length = len(text)
            text = [model[word] for word in text]
            text = np.array(text)
            text = text.sum(axis=0) / text_length
            corpus.append(text)
        corpus = np.array(corpus)
        return Normalizer().fit_transform(corpus)
    elif vectorization_method == 'bert':
        tokenizer = model[0]
        model = model[1]
        corpus = []
        for text in tqdm(texts):
            encoded_input = tokenizer(text, padding=True, truncation=True, max_length=24, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            text = cls_pooling(model_output, encoded_input['attention_mask'])
            text = np.array(text.reshape(-1, ))
            corpus.append(text)
        corpus = np.array(corpus)
        return Normalizer().fit_transform(corpus)
    elif vectorization_method == 'bm25':
        corpus = model.transform(texts)
        return corpus
    elif vectorization_method == 'tfidf':
        corpus = model.transform(texts)
        return corpus
    elif vectorization_method == 'count':
        corpus = model.transform(texts)
        corpus = Normalizer().fit_transform(corpus)
        return corpus


def get_inverse_index(corpus, vectorization_method):
    if vectorization_method == 'fasttext':
        path = input('Скачайте araneum_none_fasttextcbow_300_5_2018, распакуйте и укажите путь до'
                     'araneum_none_fasttextcbow_300_5_2018.model: ')
        path = os.path.join(path, 'araneum_none_fasttextcbow_300_5_2018.model')
        model = KeyedVectors.load(path)
        vectorized_corpus = transform_text(corpus, vectorization_method, model)
        return vectorized_corpus, model

    elif vectorization_method == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(r"sberbank-ai/sbert_large_nlu_ru")
        model = AutoModel.from_pretrained(r"sberbank-ai/sbert_large_nlu_ru")
        vectorized_corpus = transform_text(corpus, vectorization_method, (tokenizer, model))
        return vectorized_corpus, (tokenizer, model)

    elif vectorization_method == 'bm25':
        count_vectorizer = CountVectorizer().fit(corpus)
        tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

        x_count_vec = transform_text(corpus, vectorization_method, count_vectorizer)
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
        vectorized_corpus = sparse.csr_matrix((values, (rows, columns)))
        return vectorized_corpus, count_vectorizer
    elif vectorization_method == 'tfidf':
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2').fit(corpus)
        vectorized_corpus = transform_text(corpus, vectorization_method, tfidf_vectorizer)
        return vectorized_corpus, tfidf_vectorizer
    elif vectorization_method == 'count':
        count_vectorizer = CountVectorizer().fit(corpus)
        vectorized_corpus = transform_text(corpus, vectorization_method, count_vectorizer)
        return vectorized_corpus, count_vectorizer


def calculate_metric(vectorized_corpus, vectorized_queries):
    result = vectorized_corpus.dot(vectorized_queries.T).T
    if not isinstance(result, np.ndarray):
        result = result.toarray()
    result_args = result.argsort()
    total_metric = 0
    for i in range(result.shape[0]):
        args = result_args[i][::-1][:5]
        if i in args.tolist():
            total_metric += 1
    total_metric = total_metric / result.shape[0]
    return total_metric


def search(vectorization_method, questions, answers):
    # векторизация корпуса и запросов
    vectorized_corpus, vectorizer = get_inverse_index(answers, vectorization_method)
    vectorized_queries = transform_text(questions, vectorization_method, vectorizer)

    # подсчет метрики
    metric = calculate_metric(vectorized_corpus, vectorized_queries)
    return metric


def total_scores():
    questions, answers = get_corpus()
    print('Тексты проходят предобработку: ')
    morph = pymorphy2.MorphAnalyzer()
    russian_stopwords = stopwords.words("russian")
    russian_stopwords = [morph.parse(word)[0].normal_form for word in russian_stopwords]
    questions_preprocessed = [preprocessing(question, morph, russian_stopwords) for question in tqdm(questions)]
    answers_preprocessed = [preprocessing(answer, morph, russian_stopwords) for answer in tqdm(answers)]

    scores = pd.DataFrame({})
    scores['method'] = ['count', 'tfidf', 'bm25', 'fasttext', 'bert']
    scores['metric'] = np.nan
    for num, row in scores.iterrows():
        print(row['method'])
        if row['method'] == 'bert':
            metric = search(row['method'], questions, answers)
        else:
            metric = search(row['method'], questions_preprocessed, answers_preprocessed)
        scores.loc[num, 'metric'] = metric
    print(scores)


if __name__ == '__main__':
    total_scores()

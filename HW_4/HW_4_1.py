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


def get_corpus(morph, russian_stopwords, need_preprocessing=True):
    path = input('Введите путь до файла: ')
    path = os.path.join(path, 'questions_about_love.jsonl')
    with open(path, 'r', encoding='utf-8') as f:
        corpus = '[' + ', '.join(f.readlines()) + ']'
    corpus = json.loads(corpus)
    corpus = [question['answers'] for question in corpus]
    corpus = [get_max_value(answer_list) for answer_list in corpus]
    corpus = [answer for answer in corpus if pd.isnull(answer) is False][:50000]
    if need_preprocessing:
        print('Тексты проходят предобработку: ')
        preprocessed_corpus = [preprocessing(c, morph, russian_stopwords) for c in tqdm(corpus)]
    else:
        preprocessed_corpus = corpus
    return corpus, preprocessed_corpus


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
            text = np.array(text.reshape(-1,))
            corpus.append(text)
        corpus = np.array(corpus)
        return Normalizer().fit_transform(corpus)


def get_inverse_index(corpus, vectorization_method):
    if vectorization_method == 'fasttext':
        path = input('Скачайте araneum_none_fasttextcbow_300_5_2018, распакуйте и укажите путь до' 
                     'araneum_none_fasttextcbow_300_5_2018.model')
        path = os.path.join(path, 'araneum_none_fasttextcbow_300_5_2018.model')
        model = KeyedVectors.load(path)
        vectorized_corpus = transform_text(corpus, vectorization_method, model)
        return vectorized_corpus, model
    elif vectorization_method == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(r"sberbank-ai/sbert_large_nlu_ru")
        model = AutoModel.from_pretrained(r"sberbank-ai/sbert_large_nlu_ru")
        vectorized_corpus = transform_text(corpus, vectorization_method, (tokenizer, model))
        return vectorized_corpus, (tokenizer, model)


def calculate_closeness(inverse_index, query):
    rating = inverse_index.dot(query)
    rating = np.squeeze(rating)
    rating_args = rating.argsort()[::-1]
    return rating, rating_args


def search():
    #  инициализация
    morph = pymorphy2.MorphAnalyzer()
    russian_stopwords = stopwords.words("russian")
    russian_stopwords = [morph.parse(word)[0].normal_form for word in russian_stopwords]

    # получение информации
    vectorization_method = input('Введите метод векторизации (fasttext или bert): ')
    query = input('Введите запрос: ')

    if vectorization_method == 'fasttext':
        corpus, preprocessed_corpus = get_corpus(morph, russian_stopwords, need_preprocessing=True)
        inverse_index, model = get_inverse_index(preprocessed_corpus, vectorization_method)
        query = preprocessing(query, morph, russian_stopwords)
        query = transform_text([query], vectorization_method, model).T
    elif vectorization_method == 'bert':
        corpus, preprocessed_corpus = get_corpus(morph, russian_stopwords, need_preprocessing=False)
        inverse_index, model = get_inverse_index(corpus, vectorization_method)
        query = transform_text([query], vectorization_method, model).T
    # подсчет близости
    rating, rating_args = calculate_closeness(inverse_index, query)
    # вывод
    df_rating = pd.DataFrame([])
    df_rating['answer'] = np.array(corpus)[rating_args[:5]]
    df_rating['rating'] = rating[rating_args[:5]]
    print(df_rating)


if __name__ == '__main__':
    search()

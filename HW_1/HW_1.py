import zipfile
import os
import pandas as pd
import numpy as np
import pymorphy2
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer


path_to_file = r'C:\Users\VADIK\Documents\ВШЭ\final\data\IR-2021-2022\HW_1\friends-data.zip'
path = r'./'
with zipfile.ZipFile(path_to_file, 'r') as zip_ref:
    zip_ref.extractall(path)


def get_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return pd.DataFrame(data=all_files, columns=['filename'])


df = get_files(os.path.join(path, 'friends-data'))
df['text'] = np.nan
for num, row in df.iterrows():
    with open(row['filename'], 'r', encoding='utf-8-sig') as f:
        text = f.read()
    df.loc[num, 'text'] = text


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


morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")
russian_stopwords = [morph.parse(word)[0].normal_form for word in russian_stopwords]
df['preprocessed_text'] = df['text'].transform(lambda x:preprocessing(x, morph, russian_stopwords))


def create_inverse_index(corpus, vectorizer):
    X = vectorizer.fit_transform(corpus)
    inverse_index = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())
    return inverse_index


vectorizer = CountVectorizer(analyzer='word')
inverse_index = create_inverse_index(df['preprocessed_text'], vectorizer)

words = pd.DataFrame(inverse_index.sum().sort_values(), columns=['count']).reset_index()
words.rename(columns={'index': 'word'}, inplace=True)
min = words['count'].min()
max = words['count'].max()
print('Самые редко встречающиеся слова: \n', words[words['count'] == min])
print('Самые часто встречающиеся слова: \n', words[words['count'] == max])


def func(col):
    s = 0
    for i in col:
        if i != 0:
            s += 1
    return s


words_boolean = pd.DataFrame(inverse_index.apply(lambda x: func(x)), columns=['boolean_count']).reset_index()
words_boolean.rename(columns={'index': 'word'}, inplace=True)
print('Слова, встречающиеся в каждом документе: \n', words_boolean[words_boolean['boolean_count'] == 165])

characters = {
    'Моника': words[words['word'].isin(['моника', 'мон', 'мона'])]['count'].sum(),
    'Рейчел': words[words['word'].isin(['рейчел', 'рейч'])]['count'].sum(),
    'Чендлер': words[words['word'].isin(['чендлер', 'чэндлер', 'чен', 'чендлера', 'чэндлера'])]['count'].sum(),
    'Фиби': words[words['word'].isin(['фиби', 'фибс'])]['count'].sum(),
    'Росс': words[words['word'].isin(['росс'])]['count'].sum(),
    'Джоуи': words[words['word'].isin(['джоуи', 'джои', 'джо', 'джоуя', 'джой'])]['count'].sum()
}
most_popular = sorted(characters.items(), key=lambda x: x[1], reverse=True)[0][0]
print('Самый популярный персонаж — {}'.format(most_popular))

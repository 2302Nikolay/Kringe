# Используется мешок слов + логическая регрессия
import pandas as pd
import numpy as np
import pymorphy3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter

# Загрузка данных из nltk для токенизации и список стоп-слов
nltk.download('punkt')
nltk.download('stopwords')

max_words = 10000
random_state = 42

banks = pd.read_csv('banks.csv', sep='\t', index_col='idx')

def preprocess(text, stop_words, punctuation_marks, morph):
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                preprocessed_text.append(lemma)
    return preprocessed_text

punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»', ';', '–', '--']
stop_words = stopwords.words("russian")
morph = pymorphy3.MorphAnalyzer()

banks['Preprocessed_texts'] = banks.apply(lambda row: preprocess(row['Text'], punctuation_marks, stop_words, morph), axis=1)

words = Counter()

for txt in banks['Preprocessed_texts']:
    words.update(txt)

# Словарь, отображающий слова в коды
word_to_index = dict()
# Словарь, отображающий коды в слова
index_to_word = dict()

for i, word in enumerate(words.most_common(max_words - 2)):
    word_to_index[word[0]] = i + 2
    index_to_word[i + 2] = word[0]

def text_to_sequence(txt, word_to_index):
    seq = []
    for word in txt:
        index = word_to_index.get(word, 1) # 1 означает неизвестное слово
        # Неизвестные слова не добавляем в выходную последовательность
        if index != 1:
            seq.append(index)
    return seq

banks['Sequences'] = banks.apply(lambda row: text_to_sequence(row['Preprocessed_texts'], word_to_index), axis=1)

mapping = {'Negative': 0, 'Positive': 1}

banks.replace({'Score': mapping}, inplace=True)

train, test = train_test_split(banks, test_size=0.2)

x_train_seq = train['Sequences']
y_train = train['Score']

x_test_seq = test['Sequences']
y_test = test['Score']

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for index in sequence:
            results[i, index] += 1.
    return results

x_train = vectorize_sequences(x_train_seq, max_words)
x_test = vectorize_sequences(x_test_seq, max_words)

x_train[0][:100]

len(x_train[0])

lr = LogisticRegression(random_state=random_state, max_iter=500)

lr.fit(x_train, y_train)

lr.score(x_test, y_test)

positive_text = """Банк мне не понравислся. Отношение сотрудников хуже некуда. Складывалось впечатление, что ты не нужен. На все вопросы отвечали агрессивно, времени пришлось потратить очень много. Не рекомендую!
"""

positive_preprocessed_text = preprocess(positive_text, stop_words, punctuation_marks, morph)
positive_seq = text_to_sequence(positive_preprocessed_text, word_to_index)
positive_bow = vectorize_sequences([positive_seq], max_words)
positive_bow[0][0:100]
result = lr.predict(positive_bow)

print(result)
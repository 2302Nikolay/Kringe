# Импорт необходимых библиотек и модулей
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

# Задание максимального количества слов и случайного состояния для воспроизводимости результатов
max_words = 1000
random_state = 42


# Загрузка данных из файла 'banks.csv' и установка индекса
banks = pd.read_csv('your_dataset.csv', sep=',')

# Определение функции предобработки текста
def preprocess(text, stop_words, punctuation_marks, morph):
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                preprocessed_text.append(lemma)
    return preprocessed_text


# Задание списка пунктуационных знаков и загрузка стоп-слов для русского языка
punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»', ';', '–', '--']
stop_words = stopwords.words("english")
morph = pymorphy3.MorphAnalyzer()


# Применение функции предобработки ко всем текстам в датафрейме
banks['Preprocessed_texts'] = banks.apply(lambda row: preprocess(row['review'], punctuation_marks, stop_words, morph), axis=1)

# Использование Counter для подсчета частоты слов
words = Counter()

for txt in banks['Preprocessed_texts']:
    words.update(txt)


# Создание словарей для отображения слов в индексы и наоборот
word_to_index = dict()
index_to_word = dict()

for i, word in enumerate(words.most_common(max_words - 2)):
    word_to_index[word[0]] = i + 2
    index_to_word[i + 2] = word[0]


# Функция для преобразования текста в последовательность индексов
def text_to_sequence(txt, word_to_index):
    seq = []
    for word in txt:
        index = word_to_index.get(word, 1)  # 1 означает неизвестное слово
        if index != 1:
            seq.append(index)
    return seq


# Применение функции text_to_sequence ко всем текстам в датафрейме
banks['Sequences'] = banks.apply(lambda row: text_to_sequence(row['Preprocessed_texts'], word_to_index), axis=1)

# Замена меток классов 'Negative' и 'Positive' числовыми значениями 0 и 1
mapping = {'negative': 0, 'positive': 1}
banks.replace({'sentiment': mapping}, inplace=True)


# Разделение данных на обучающий и тестовый наборы
train, test = train_test_split(banks, test_size=0.2)

# Получение входных данных и меток классов для обучающего и тестового наборов
x_train_seq = train['Sequences']
y_train = train['sentiment']

x_test_seq = test['Sequences']
y_test = test['sentiment']


# Функция для векторизации последовательностей
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for index in sequence:
            results[i, index] += 1.
    return results


# Векторизация обучающего и тестового наборов
x_train = vectorize_sequences(x_train_seq, max_words)
x_test = vectorize_sequences(x_test_seq, max_words)

# Инициализация и обучение модели логистической регрессии
lr = LogisticRegression(random_state=random_state, max_iter=500)
lr.fit(x_train, y_train)


# Оценка точности модели на тестовом наборе
lr.score(x_test, y_test)

def predict_sentiment(text, model=lr, max_words=max_words):
    # Предобработка текста
    preprocessed_text = preprocess(text, stop_words, punctuation_marks, morph)
    # Преобразование текста в последовательность индексов
    sequence = text_to_sequence(preprocessed_text, word_to_index)
    # Векторизация последовательности
    bow = vectorize_sequences([sequence], max_words)
    # Предсказание с использованием модели
    result = model.predict(bow)
    
    # Возвращение результата
    return "Положительный" if result == 1 else "Негативный"


import numpy as np

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical

with open('text', 'r', encoding='utf-8') as f:
    texts = f.read()
    texts = texts.replace('\ufeff', '')

print(texts)
maxWordsCount = 1000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts([texts])
dist = list(tokenizer.word_counts.items())
print(dist)
data = tokenizer.texts_to_sequences([texts])
print(len(data[0]))
res = to_categorical(data[0], num_classes=maxWordsCount)
print(res.shape)

inp_words = 3
n = res.shape[0] - inp_words

X = np.array([res[i:i+inp_words, :] for i in range(n)])
Y = res[inp_words:]

model = Sequential()
model.add(Input((inp_words, maxWordsCount)))
model.add(SimpleRNN(128, activation='tanh', return_sequences=True))  # return_sequences - нужен чтобы использовать несколько слоев рнн подряд
model.add(SimpleRNN(64, activation='tanh'))
model.add(Dense(maxWordsCount, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
history = model.fit(X, Y, batch_size=32, epochs=100)


def buildPhrase(texts, str_len=20):
    res = texts
    data = tokenizer.texts_to_sequences([texts])[0]
    for i in range(str_len):
        x = to_categorical(data[i:i+inp_words], num_classes=maxWordsCount)
        inp = x.reshape(1, inp_words, maxWordsCount)

        pred = model.predict(inp)
        indx = pred.argmax(axis=1)[0]

        data.append(indx)

        res += ' ' + tokenizer.index_word[indx]
    return res

res = buildPhrase('позитив добавляет годы')
print(res)



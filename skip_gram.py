from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.text import text_to_word_sequence

from keras.layers import Input
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot

from keras.models import Model
from keras.utils import plot_model

import os
import numpy as np

# read data
DATA_DIR = '/Users/shuchendu/Downloads'

with open(os.path.join(DATA_DIR, 'text8')) as f:
    text = f.read()
f.close()

# use 10 percent to text for training
text = text_to_word_sequence(text)
text = text[: int(len(text) / 100)]
print(len(text))

# tokenize data
tkn = Tokenizer()
tkn.fit_on_texts(text)
print('tokenize data')

word2idx = tkn.word_index
idx2word = {v: k for k, v in word2idx.items()}
print(len(word2idx))

# skip grams
idx_seq = [word2idx[w] for w in text]

data, labels = skipgrams(idx_seq, len(word2idx))
print('skip grams')

# word and context data
word_data = np.array([d[0] for d in data]).reshape(-1, 1)
context_data = np.array([d[1] for d in data]).reshape(-1, 1)
labels = np.array(labels).reshape(-1, 1, 1)
print('shapes: {}, {}, {}'.format(word_data.shape, context_data.shape, labels.shape))

# define network
word_input = Input(shape=(1, ))
emb_word = Embedding(len(word2idx) + 1,
                     output_dim=300,
                     embeddings_initializer='glorot_uniform',
                     input_length=1)(word_input)

context_input = Input(shape=(1, ))
emb_context = Embedding(len(word2idx) + 1,
                        output_dim=300,
                        embeddings_initializer='glorot_uniform',
                        input_length=1)(context_input)

merge = dot([emb_word, emb_context], axes=2)
dense = Dense(1, activation='sigmoid')(merge)
sg_mdl = Model(inputs=[word_input, context_input], outputs=dense)
plot_model(model=sg_mdl, to_file='sg.png', show_shapes=True)
sg_mdl.compile(loss='mse', optimizer='adam')

# fit data
sg_mdl.fit([word_data, context_data], labels, batch_size=256, epochs=1)

# save model
sg_mdl.save('sg_100.h5')



# coding: utf-8

# In[14]:

from flask import Flask,render_template,request
import collections
import cgi


import helper
import numpy as np
import project_tests as tests
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential,load_model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional,SimpleRNN,Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from IPython.display import display
import ipywidgets as widgets
import click

app=Flask(__name__)


@app.route('/')
def my_form():
    return render_template("front_end.html")
#@app.route("/front_end.html")
#def get_result():
 #   return french
@app.route('/submit', methods=['POST'])
def submit():
    #return 'You entered: {}'.format(request.form['txtSearch'])
    input_text=request.form['txtSearch']
    y_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    model = load_model('savedtrain_data.h5')
    input_text = [english_tokenizer.word_index[word] for word in input_text.split()]

    input_text = pad_sequences([input_text], maxlen=preproc_english_sentences.shape[-1], padding='post')
    input_texts = np.array([input_text[0], preproc_english_sentences[0]])
    predictions = model.predict(input_texts, len(input_texts))

    output = ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])
    french = output.replace('<PAD>', '')
    return french

# Load English data
english_sentences = helper.load_data('small_vocab_en')
# Load French data
french_sentences = helper.load_data('small_vocab_fr')

print('Dataset Loaded')


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(x)
    tokenized_data=tokenizer.texts_to_sequences(x)
    return tokenized_data,tokenizer


tests.test_tokenize(tokenize)

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    padded_sequences=pad_sequences(x,maxlen=length,padding='post')
    return padded_sequences
tests.test_pad(pad)


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk
input_text=''
@app.route('/input',methods=['POST'])

def input():
    input_text =request.form['txtSearch']
    return '%s'


print(input_text)



preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(english_sentences, french_sentences)


max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)



#def input_function():
    #input_text = input("Enter the text to be translated") 
    #preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
    #preprocess(input_text, french_sentences)
    

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


print('`logits_to_text` function loaded.')



#button.on_click(on_button_click)
if __name__=="__main__":
    app.run(debug=True)




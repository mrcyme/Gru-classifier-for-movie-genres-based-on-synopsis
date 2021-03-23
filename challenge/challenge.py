"""
This module provides an method to predict movie genre based on synopsis.

It is a keras implementation of the multinomial GRU approach of paper
.. [1] Hoang, Quan. (2018). Predicting Movie Genres Based on Plot Summaries.

The glove pretrain vectors can be downloaded at:
https://nlp.stanford.edu/projects/glove/
Embedding to use for the code is the 100d embedding

It provides to train and a prodict method :
Train : Endpoint used for training a GRU model on a dataset containing movie
synopsis and movie genres
Predict : Endpoint used for predicting movie genre based on a dataset
containing movie synopsis.

"""
import pandas as pd
import numpy as np
import re
import json
from tensorflow.keras.layers import Dense, Embedding, GRU
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

EARLY_STOPPING = EarlyStopping(monitor='val_loss',
                               mode='min',
                               restore_best_weights=True)
config = {
    "MAX_WORDS": 80000,
    "EMB_DIM": 100,
    "NUMBER_OF_GENRES": 19,
    "BATCH_SIZE": 256,
    "N_EPOCHS": 10,
    "VECTOR_LEN": 200,
    "DROUPOUT_RATE": 0.6,
    "HIDDEN_DIM": 128,
    "LEARNING_RATE": 0.002}

genre_dic = {
    "Action": 0,
    "Adventure": 1,
    "Animation": 2,
    "Children": 3,
    "Comedy": 4,
    "Crime": 5,
    "Documentary": 6,
    "Drama": 7,
    "Fantasy": 8,
    "Film-Noir": 9,
    "Horror": 10,
    "IMAX": 11,
    "Musical": 12,
    "Mystery": 13,
    "Romance": 14,
    "Sci-Fi": 15,
    "Thriller": 16,
    "War": 17,
    "Western": 18}


def generate_glove_embedding(tokenizer):
    """Generate glove embedding matrix.

    Uses the glove's pretrained vectors to generate an embedding matric

    :param tokenizer : tokenizer used to convert synopsis to int sequences
    :return : The embedding matrix used in the embedding layer of the GRU
    """
    embeddings_dictionary = dict()
    with open('challenge/glove.6B.100d.txt', encoding="utf8") as glove_file:
        for line in glove_file:
            records = line.split()
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embeddings_dictionary[records[0]] = vector_dimensions
    embedding_matrix = np.zeros((config["MAX_WORDS"], 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix


def clean_text(text):
    """Text Preprocessing function.

    Remove apostrophe from text then get rid of all non letters caracters
    and convert to lowercase

    :param text : the text to clean
    :return : the cleaned text
    """
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    stopWords = set(stopwords.words('english'))
    text = ' '.join([w for w in text.split() if w not in stopWords])
    return text


def preprocess_synopsis(df, tokenizer=None):
    """Convert synopsis to padded integer sequences.

    Generate a tokenizer or use a given tokenizer to produce
    integer sequences. The sequences are then padded.

    :param df : Dataframe containing a synopsis column
    :param tokenizer : tokenizer used to convert text to integer sequence
    """
    x = df['synopsis'].apply(lambda x: clean_text(x)).to_numpy()
    if not tokenizer:
        tokenizer = text.Tokenizer(num_words=config["MAX_WORDS"])
        tokenizer.fit_on_texts(x)
        with open('tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
    x_seq = tokenizer.texts_to_sequences(x)
    x_pad = sequence.pad_sequences(x_seq, maxlen=config["VECTOR_LEN"])
    return x_pad, tokenizer


def preprocess_genres(df):
    """Convert genres to multinomial one hot representation.

    A list of genre is converted to a vector of length equal
    to the total number of genre.
    The entry corresponding to a certain genre equals
    one divided by the number of genre in the list if the genre is present
    and zero otherwise.

    :param df : Dataframe containing a genres column
    :return : one hot encodding of the genres
    """
    genres = df['genres'].apply(lambda x: x.split(" ")).to_numpy()
    genre_dic = json.load(open("challenge/genre_dic.json", "r"))
    y_one_hot = np.zeros((len(genres), config["NUMBER_OF_GENRES"]))
    for i in range(len(genres)):
        for g in genres[i]:
            y_one_hot[i][genre_dic[g]] = 1
    return np.array([[x / np.sum(row) for x in row] for row in y_one_hot])


def probas_to_top_five(y_prob):
    """Convert genre probabilities to a genre list.

    Convert the vector of redicted_genres probabilities to a list
    containing the five genre with highest probabilities.

    :param y_prodb : vector of genre probabilities
    :return : list of the five genre with highest probabilities
    """
    indices = np.argsort(-y_prob)[:5]
    genre_dic = json.load(open("challenge/genre_dic.json", "r"))
    inv_genre_dic = {v: k for k, v in genre_dic.items()}
    top_five_genre = [inv_genre_dic[ind] for ind in indices]
    return top_five_genre


def generate_gru(tokenizer):
    """Generate GRU model.

    Generate the GRU model with the defined config.

    :param : tokenizer : tokenizer used to convert text to integer sequence
    :return : compiled GRU model
    """
    embedding_matrix = generate_glove_embedding(tokenizer)
    model_gru = Sequential()
    model_gru.add(Embedding(config["MAX_WORDS"],
                            config["EMB_DIM"],
                            weights=[embedding_matrix],
                            trainable=True))
    model_gru.add(GRU(config["HIDDEN_DIM"],
                  dropout=config["DROUPOUT_RATE"],
                  return_sequences=False))
    model_gru.add(Dense(config["NUMBER_OF_GENRES"], activation='softmax'))
    optimizer = Adam(learning_rate=config["LEARNING_RATE"])
    model_gru.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy'])
    return model_gru


def train(train_file):
    """
    Training endpoint.
    """
    df_movies = pd.read_csv(train_file)
    x_train, tokenizer = preprocess_synopsis(df_movies)
    y_train = preprocess_genres(df_movies)
    model = generate_gru(tokenizer)
    model.fit(x_train, y_train, batch_size=config["BATCH_SIZE"],
              validation_split=0.1, epochs=config["N_EPOCHS"],
              callbacks=[EARLY_STOPPING])
    model.save_weights("gru.h5")


def predict(test_file):
    """
    Prediction endpoint.
    """
    df_movies = pd.read_csv(test_file)
    with open('tokenizer.json') as f:
        tokenizer = text.tokenizer_from_json(json.load(f))
    x_test, _ = preprocess_synopsis(df_movies, tokenizer=tokenizer)
    model = generate_gru(tokenizer)
    model.load_weights("gru.h5")
    y_list = [probas_to_top_five(y) for y in model.predict(x_test)]
    data = df_movies.drop(["synopsis", "year"], axis=1)
    data["predicted_genres"] = np.array([" ".join(e) for e in y_list])
    return data

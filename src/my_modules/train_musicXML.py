from __future__ import print_function

import glob
import os
import random
import sys

import music21 as m21
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.models import Sequential, load_model
from tqdm import tqdm

device_lib.list_local_devices()
import shutil
import sys

from music21 import *


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    print()
    print("------- Generating text after Epoch: %d" % epoch)
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2]:  # ここは0.2のみ？
        print("--------diversity:", diversity)

        generated = ""
        sentence = text[start_index : start_index + maxlen]
        # sentenceはリストなので文字列へ変換
        generated += "".join(sentence)
        # print(sentence)

        # print('--------- Generating with seed:"' + "".join(sentence) + '"')
        # sys.stdout.write(generated)

        for i in range(100):
            x_pred = np.zeros((1, maxlen, len(char_indices)))  # len(chars) → len(char_indices)に変更
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0

            preds = model.predict(x_pred, verbose=0)[0]
            # print('preds:' + str(len(preds)))
            # print('preds:' + str(preds))
            next_index = sample(preds, diversity)
            # print('next_index:' + str(next_index))
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:]
            sentence.append(next_char)

            # sys.stdout.write(next_char)
            # sys.stdout.flush()
        print()

if __name__ == "__main__":
    args = sys.argv
    senti = args[1]
    epochs = 100
    batch_size = 128

    xmlpath = "./assets/musicxml/" + senti
    model_save_path = f"./static/models/{senti}.keras"
    make_model = True
    music_keys = "C"

    text = []

    xmls = glob.glob(xmlpath + "/*")
    for x in tqdm(xmls):
        piece = m21.converter.parse(x)

        for trans_key in music_keys:
            k = piece.analyze("key")
            trans = trans_key
            i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(trans))
            trans_piece = piece.transpose(i)
            for n in trans_piece.flat.notesAndRests:
                if type(n) == m21.note.Note:
                    text.append(str(n.name) + "_" + str(n.duration.quarterLength) + " ")
                elif type(n) == m21.chord.Chord:
                    pitches = "~".join([pitche.name for pitche in n.pitches])
                    text.append(str(pitches) + "_" + str(n.duration.quarterLength) + " ")

    print("--------- start LSTM")
    chars = text
    count = 0
    char_indices = {}
    indices_char = {}
    for word in chars:
        if word not in char_indices:
            char_indices[word] = count
            count += 1
            # print(count, word)
    indices_char = dict([(value, key) for (key, value) in char_indices.items()])
    maxlen = 10
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i : i + maxlen])
        next_chars.append(text[i + maxlen])
    print("nb sequences:", len(sentences))

    print("sentence:")
    print("Vectorization...")
    x = np.zeros((len(sentences), maxlen, len(char_indices)), dtype=bool)
    y = np.zeros((len(sentences), len(char_indices)), dtype=bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

    print("Build model...")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(maxlen, len(char_indices))))
    model.add(tf.keras.layers.LSTM(500))
    model.add(tf.keras.layers.Dense(len(char_indices), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", run_eagerly=True)
    
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    es_cb = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, verbose=0, mode="auto")
    model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[es_cb, print_callback])
    model.save(model_save_path)
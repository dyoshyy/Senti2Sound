from __future__ import print_function

import glob
import os
import random
import shutil
from fractions import Fraction

import music21 as m21
import numpy as np
from music21 import *

# from tensorflow.python.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

MAX_LENGTH = 10  # 時系列を何個ずつに分けて学習するか

# モデル生成準備
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)


def make_melody(model, text, char_indices, indices_char, length=200):
    start_index = random.randint(0, len(text) - MAX_LENGTH - 1)
    for diversity in [0.2]:  # ここは0.2のみ？

        generated = ""
        sentence = text[start_index : start_index + MAX_LENGTH]
        # sentenceはリストなので文字列へ変換
        generated += "".join(sentence)

        for i in range(length):
            x_pred = np.zeros((1, MAX_LENGTH, len(char_indices)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:]
            sentence.append(next_char)

    return generated


def process_xml_files(xmlpath, music_keys):
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
    return text


def generate(senti, length, inst_id):

    xmlpath = f"/workspaces/Senti2Sound/src/assets/musicxml/{senti}"
    model_path = f"/workspaces/Senti2Sound/src/static/models/{senti}.keras"
    music_keys = "C"
    print("Current Directory:",os.getcwd())
    # テキストの生成
    text = process_xml_files(xmlpath, music_keys)

    # ここからLSTM
    chars = text
    count = 0

    char_indices = {}  # 辞書
    for word in chars:
        if not word in char_indices:
            char_indices[word] = count
            count += 1
    # 逆引き辞書を辞書から作成する
    indices_char = dict([(value, key) for (key, value) in char_indices.items()])
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - MAX_LENGTH, step):
        sentences.append(text[i : i + MAX_LENGTH])
        next_chars.append(text[i + MAX_LENGTH])

    # モデルがある場合は読み込む,なければ学習
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        # model = load_model(model_path,compile=False)
        # model.load_weights(model_weights_path)
    else:
        print("--------Model does not exist----------")

    melo_sentence = make_melody(model, text, char_indices, indices_char, length)
    # メロディをmusicXMLに変換する
    meas = m21.stream.Stream()
    meas.append(m21.meter.TimeSignature("4/4"))

    instr = instrument.instrumentFromMidiProgram(inst_id)
    meas.insert(instr)

    melo = melo_sentence.split()
    for m in melo:
        ptches, dist = m.split("_")
        dist = Fraction(dist)
        if ptches == "rest":
            n = m21.note.Rest(quarterLength=float(dist))
        elif "~" in ptches:
            ptche_list = ptches.split("~")
            n = m21.chord.Chord(ptche_list, quarterLenght=float(dist))
        else:
            n = m21.note.Note(ptches, quarterLength=float(dist))

        meas.append(n)

    meas.makeMeasures(inPlace=True)
    meas.write("midi", str(senti) + ".mid")
    shutil.move(f"{senti}.mid", f"/workspaces/Senti2Sound/src/static/generated/{senti}.mid")

from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras import optimizers 
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
import keras
import random
import sys
import io
from tqdm import tqdm
import music21 as m21
import os
import glob
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from music21 import *
from midi2audio import FluidSynth
import fluidsynth
import pretty_midi
import sounddevice as sd
import scipy
from scipy import io
from scipy.io import wavfile
from pprint import pprint


us = environment.UserSettings()
#us.create() #first time only
us['lilypondPath'] = 'C:/LilyPond/usr/bin/lilypond.exe'
us['musescoreDirectPNGPath'] = 'C:/Program Files/MuseScore4/bin/MuseScore4.exe'
us['musicxmlPath'] = 'C:/Program Files/MuseScore4/bin/MuseScore4.exe'

DS = os.sep
bs = os.path.dirname(__file__) + DS
xmlpath = bs + 'musicxml_simple' + DS

#bs = '/content/drive/MyDrive'
#xmlpath = '/content/drive/MyDrive/musicXML'

model_weights_path = 'melo_model' + 'w.hdf5'
model_save_path = 'melo_model' + '.hdf5'
make_model = True
# music_keys = ('C', 'D', 'E', 'F', 'F#', 'G', 'A', 'B')
music_keys = ('C')
# テキストの生成
text = []

# フォルダ内のxmlファイルを取得する
xmls = glob.glob(xmlpath + "/*")
for x in tqdm(xmls):
    # xmlを読み込む
    piece = m21.converter.parse(x)

    for trans_key in music_keys:
        k = piece.analyze('key')
        # 主音を合わせる
        trans = trans_key
        i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(trans))
        trans_piece = piece.transpose(i)
        for n in trans_piece.flat.notesAndRests: 
            if type(n) == m21.note.Note:
                text.append(str(n.name) + '_' +str(n.duration.quarterLength) + ' ')
            #elif type(n) == m21.chord.Chord:
                #print(p for p in n.notes)
                #text.append(str(m.name)+'_'+str(m.duration.quarterLength)+' ' for m in n.normalOrder)


# ここからLSTM
print('--------- start LSTM')
chars = text
count = 0
char_indices = {}  # 辞書
indices_char = {}  # 逆引き辞書
for word in chars:
    if not word in char_indices:
        char_indices[word] = count
        count += 1
        print(count, word)  # 登録した単語を表示
# 逆引き辞書を辞書から作成する
indices_char = dict([(value, key) for (key, value) in char_indices.items()])
maxlen = 5
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
'''
# モデルのファイルがある場合は読み取る
if (os.path.exists(model_save_path) and os.path.exists(model_weights_path)):
    print('-----------read Model')
    model = load_model(model_save_path, compile=False)
    model.load_weights(model_weights_path)
'''
pprint(sentences)
# モデル生成準備
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		x[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1

#print('--------------------')
#pprint(x)
#pprint(y)

# モデル生成
print('Build model...')
model = Sequential()
model.add(LSTM(500, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
#optimizer = RMSprop(learning_rate=0.01)
#opt = optimizers.
model.compile(loss='categorical_crossentropy',
            #optimizer=opt, 
            run_eagerly=True)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    print()
    print('------- Generating text after Epoch: %d' % epoch)
    start_index = random.randint(0, len(text) - maxlen - 1)
    start_index = 0  # テキストの最初からスタート
    for diversity in [0.2]:  # ここは0.2のみ？
        print('--------diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        # sentenceはリストなので文字列へ変換
        generated += ''.join(sentence)
        print(sentence)

        print('--------- Generating with seed:"' + "".join(sentence) + '"')
        sys.stdout.write(generated)

        for i in range(100):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:]
            sentence.append(next_char)

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


def on_train_end(logs):
    print('----- saving model...')
    model.save_weights(model_weights_path)
    model.save(model_save_path)


def make_melody(length=200):
    start_index = random.randint(0, len(text) - maxlen - 1)
    # start_index = 0 #テキストの最初からスタート

    print(start_index)
    for diversity in [0.2]:  # ここは0.2のみ？
        print('--------diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        # sentenceはリストなので文字列へ変換
        generated += ''.join(sentence)
        print(sentence)

        print('--------- Generating with seed:"' + "".join(sentence) + '"')
        sys.stdout.write(generated)

        for i in range(length):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:]
            sentence.append(next_char)

            #sys.stdout.write(next_char)
            #sys.stdout.flush()
        print()

    return generated

#モデルがある場合は読み込む　なければ学習
if (os.path.exists(model_save_path) and os.path.exists(model_weights_path)):
    print('-----------read Model')
    model = load_model(model_save_path, compile=False)
    model.load_weights(model_weights_path)
else:
	print_callbak = LambdaCallback(on_epoch_end=on_epoch_end, on_train_end=on_train_end)
	es_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=7, verbose=0, mode='auto')
	model.fit(x, y, batch_size=128, epochs=300, callbacks=[es_cb,print_callbak])

print('-------print score')
melo_sentence = make_melody(60)
print(melo_sentence)
# メロディをmusicXMLに変換する
meas = m21.stream.Stream()
meas.append(m21.meter.TimeSignature('4/4'))
melo = melo_sentence.split()
for m in melo:
    ptch, dist = m.split('_')
    if (ptch == 'rest'):
        n = m21.note.Rest(quarterLength=float(dist))
    else:
        n = m21.note.Note(ptch, quarterLength=float(dist))

    meas.append(n)

    # print(note_dt)

meas.makeMeasures(inPlace=True)
#meas.show('mxl', addEndTimes=True)
meas.write("midi", "generated.mid")


fs = FluidSynth(sound_font='SGM.sf2')
input = 'generated' + '.mid'
output = 'output' + '.wav'
fs.midi_to_audio( input , output )

#meas.show('musicxml')

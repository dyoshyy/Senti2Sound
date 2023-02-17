from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
import numpy as np
import random
import sys
from tqdm import tqdm
import music21 as m21
import os
import glob
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
from fractions import Fraction
from music21 import *
import shutil
import sys

args = sys.argv
#senti = input("input the senti:")
senti = args[1]
epochs = int(args[2])

DS = os.sep
bs = os.path.dirname(__file__) + DS
xmlpath = bs + 'musicxml\\' + senti + DS

#bs = '/content/drive/MyDrive'
#xmlpath = '/content/drive/MyDrive/musicXML'

model_weights_path = 'model_' + senti + 'w.hdf5'
model_save_path = 'model_' + senti + '.hdf5'
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
            elif type(n) == m21.chord.Chord:
                 pitches = "~".join([pitche.name for pitche in n.pitches])
                 #print(pitches)
                 text.append(str(pitches) + '_' +str(n.duration.quarterLength) + ' ')

#pprint(text)
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
maxlen = 5 #時系列を何個ずつに分けて学習するか
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

#pprint(char_indices)
#pprint(indices_char)

print('sentence:')
#pprint(sentences)
# モデル生成準備
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(char_indices)), dtype=bool) #len(chars)→len(char_indices)に変更
y = np.zeros((len(sentences), len(char_indices)), dtype=bool)         #len(chars)→len(char_indices)に変更
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
model.add(LSTM(500,input_shape=(maxlen, len(char_indices)))) #len(chars) → len(char_indices)に変更
model.add(Dense(len(char_indices), activation='softmax'))  #len(chars) → len(char_indices)に変更
#opt = tf.optimizers.SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',
            optimizer='adagrad', 
            run_eagerly=True)
#model.summary()

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
    for diversity in [0.2]:  # ここは0.2のみ？
        print('--------diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        # sentenceはリストなので文字列へ変換
        generated += ''.join(sentence)
        #print(sentence)

        print('--------- Generating with seed:"' + "".join(sentence) + '"')
        sys.stdout.write(generated)

        for i in range(100):
            x_pred = np.zeros((1, maxlen, len(char_indices))) #len(chars) → len(char_indices)に変更
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            #print('preds:' + str(len(preds)))
            #print('preds:' + str(preds))
            next_index = sample(preds, diversity)
            #print('next_index:' + str(next_index))
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

'''
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
            x_pred = np.zeros((1, maxlen, len(char_indices))) #len(chars) → len(char_indices)に変更
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
'''
    

#モデルがある場合は読み込む　なければ学習
if (os.path.exists(model_save_path) and os.path.exists(model_weights_path)):
    print('-----------read Model')
    model = load_model(model_save_path,compile=False)
    model.load_weights(model_weights_path)
else:
	print_callbak = LambdaCallback(on_epoch_end=on_epoch_end, on_train_end=on_train_end)
	es_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7, verbose=0, mode='auto')
	model.fit(x, y, batch_size=128, epochs=epochs, callbacks=[es_cb,print_callbak])


print("train comleted")

shutil.move(model_weights_path, r'models\\'+str(senti) + "\\" + model_weights_path)
shutil.move(model_save_path, r'models\\'+str(senti) + "\\" + model_save_path)

'''
print('-------print score')
melo_sentence = make_melody(60)
print(melo_sentence)
# メロディをmusicXMLに変換する
meas = m21.stream.Stream()
meas.append(m21.meter.TimeSignature('4/4'))
melo = melo_sentence.split()
for m in melo:
    ptches, dist = m.split('_')
    dist = Fraction(dist)
    if (ptches == 'rest'):
        n = m21.note.Rest(quarterLength=float(dist))
    elif '~' in ptches:
        ptche_list = ptches.split('~')
        n = m21.chord.Chord(ptche_list,quarterLenght=float(dist))
    else:
        n = m21.note.Note(ptches, quarterLength=float(dist))

    meas.append(n)

    # print(note_dt)

meas.makeMeasures(inPlace=True)
#meas.show('mxl', addEndTimes=True)
meas.write("midi", "generated.mid")

meas.show('musicxml')
'''
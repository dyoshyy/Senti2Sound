from __future__ import print_function
from tensorflow.python.keras.models import load_model
import numpy as np
import random
from tqdm import tqdm
import music21 as m21
from music21 import *

import os
import glob
import shutil

from fractions import Fraction

def generate(senti,length,inst_id):

    DS = os.sep
    bs = os.path.dirname(__file__) + DS

    #xmlpath = bs + 'musicxml\\' + str(senti) + DS
    xmlpath = os.path.join("musicxml", str(senti))

    '''
    model_path_base = bs + 'models\\' + str(senti) + '\\model_' + str(senti)
    model_weights_path = model_path_base + 'w.hdf5' 
    model_save_path = model_path_base + '.hdf5' 
    '''
    model_name = "model_" + str(senti)
    model_weights_path = os.path.join("models", model_name + "w.hdf5")
    model_save_path = os.path.join("models", model_name + ".hdf5")
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

    # ここからLSTM
    chars = text
    count = 0

    char_indices = {}  # 辞書
    indices_char = {}  # 逆引き辞書
    for word in chars:
        if not word in char_indices:
            char_indices[word] = count
            count += 1
    # 逆引き辞書を辞書から作成する
    indices_char = dict([(value, key) for (key, value) in char_indices.items()])
    maxlen = 10 #時系列を何個ずつに分けて学習するか
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    # モデル生成準備
    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def make_melody(length=200):
        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2]:  # ここは0.2のみ？

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            # sentenceはリストなので文字列へ変換
            generated += ''.join(sentence)

            for i in range(length):
                x_pred = np.zeros((1, maxlen, len(char_indices))) 
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:]
                sentence.append(next_char)


        return generated

    #モデルがある場合は読み込む　なければ学習
    if (os.path.exists(model_save_path) and os.path.exists(model_weights_path)):
        model = load_model(model_save_path,compile=False)
        model.load_weights(model_weights_path)
    else:
        print('--------Model does not exist----------')

    melo_sentence = make_melody(length)
    print(melo_sentence)
    # メロディをmusicXMLに変換する
    meas = m21.stream.Stream()
    meas.append(m21.meter.TimeSignature('4/4'))

    #instr = instrument.Trumpet()
    instr = instrument.instrumentFromMidiProgram(inst_id)
    meas.insert(instr)
    #instr.midiProgram = 56


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

    meas.makeMeasures(inPlace=True)
    meas.write("midi", str(senti) + ".mid")
    shutil.move(str(senti)+".mid", r'static\\generated\\'+str(senti)+'.mid')
    #meas.show('musicxml')

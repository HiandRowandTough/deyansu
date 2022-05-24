#1. モジュールのインポート

import numpy as np
import copy
import tensorflow as tf
import random
import re
import json
import pickle
from janome.tokenizer import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from flask_cors import CORS # <-追加
from flask import Flask, g, request


#2. 入力処理(複数文の場合は分けて、リストに格納)

#複数文を分割(「。」もしくは「?」で区別する)
def Split_Sentence(inputText):
    text_list = re.split("(?<=[。?？])", inputText)
    text_list = [x for x in text_list if x !='']
    return text_list


#3. 形態素解析

#分かち書きのメソッド
def Wakati_Method(inputText):
    t = Tokenizer(udic='userdic.csv', udic_enc='utf8', wakati=True)
    wakati_textlist = []
    for token in t.tokenize(inputText):
        wakati_textlist.append(token)
    return wakati_textlist
    
special_chars = ['<pad>', '<s>', '</s>', '<unk>']  #パディング用、bos、eos、未知語
bos_char = special_chars[1]
eos_char = special_chars[2]
oov_char = special_chars[3]


#4. インデックス変換(語尾以外はリストに格納)

#w2i.pickleからw2i,i2wを取得
with open('w2i.pickle', mode='rb') as f:
    w2i = pickle.load(f)
i2w = {i: w for w, i in w2i.items()}

#インデックス変換のためのメソッド
def word2index(text, w2i):
    eoslist = []
    count = 0
    for i in range(len(text)-1, 0, -1):
        count += count
        if(count > 5):
            break
        if text[i] not in w2i:
            break
        else:
            eoslist.append(w2i[text[i]])
            del text[i]      
    return text, [1] + eoslist[::-1] + [2]


#5. モデルへ入力

#エンコーダ・デコーダクラスを定義
class EncoderDecoder(Model):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 maxlen=10):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

        self.maxlen = maxlen
        self.output_dim = output_dim

    def call(self, source, target=None, use_teacher_forcing=False):
        batch_size = source.shape[0]
        if target is not None:
            len_target_sequences = target.shape[1]
        else:
            len_target_sequences = self.maxlen

        _, states = self.encoder(source)
        
        y = tf.ones((batch_size, 1), dtype=tf.int32)
        output = tf.zeros((batch_size, 1, self.output_dim), dtype=tf.float32)

        for t in range(len_target_sequences):
            out, states = self.decoder(y, states)
            out = out[:, tf.newaxis]
            output = tf.concat([output, out], axis=1)

            if use_teacher_forcing and target is not None:
                y = target[:, t][:, tf.newaxis]
            else:
                y = tf.argmax(out, axis=-1, output_type=tf.int32)

        return output[:, 1:]

class Encoder(Model):
    def __init__(self,
                 input_dim,
                 hidden_dim):
        super().__init__()
        self.embedding = Embedding(input_dim, hidden_dim, mask_zero=True)
        self.lstm = LSTM(hidden_dim, activation='tanh',
                         recurrent_activation='sigmoid',
                         kernel_initializer='glorot_normal',
                         recurrent_initializer='orthogonal',
                         return_state=True)

    def call(self, x):
        x = self.embedding(x)
        h, state_h, state_c = self.lstm(x)
        return h, (state_h, state_c)

class Decoder(Model):
    def __init__(self,
                 hidden_dim,
                 output_dim):
        super().__init__()
        self.embedding = Embedding(output_dim, hidden_dim)
        self.lstm = LSTM(hidden_dim, activation='tanh',
                         recurrent_activation='sigmoid',
                         kernel_initializer='glorot_normal',
                         recurrent_initializer='orthogonal',
                         return_state=True)
        self.out = Dense(output_dim, kernel_initializer='glorot_normal',
                         activation='softmax')

    def call(self, x, states):
        x = self.embedding(x)
        h, state_h, state_c = self.lstm(x, states)
        y = self.out(h)

        return y, (state_h, state_c)

#ハイパーパラメータの設定、モデルの定義
input_dim = 128
hidden_dim = 128
output_dim = 128

#index2word関数の定義
def index2word(i2w, sentence):
    return [i2w[id] for id in sentence]


#6. 入力をコンバートする(メイン関数)

def convert(x):
    inputTextList = Split_Sentence(x)

    for i in range(len(inputTextList)):
        inputTextList[i] = Wakati_Method(inputTextList[i])

    index_list = ["" for x in range(len(inputTextList))]
    for i in range(len(inputTextList)):
       inputTextList[i], index_list[i] = word2index(inputTextList[i], w2i)

    index_list = pad_sequences(index_list, maxlen=5, padding='post')
    index_list = tf.convert_to_tensor(index_list, dtype=tf.int32)

    model = EncoderDecoder(input_dim, hidden_dim, output_dim)

    #保存したモデルの復元
    model.load_weights('./checkpoints/my_checkpoint')

    output = ["" for i in range(len(index_list))]

    for i in range(len(index_list)):
        preds = tf.reshape(index_list[i], [1, 5])
        preds = model(preds)
        out = tf.argmax(preds, axis=-1).numpy().reshape(-1)
        output[i] = index2word(i2w, out)
        output[i] = inputTextList[i] + [x for x in output[i] if not x in special_chars]
        output[i] = ''.join(output[i])

    return ''.join(output)

app = Flask(__name__)
CORS(app) # <-追加

@app.route('/convert/<mes>', methods=['GET'])
def get_post(mes):
  return json.dumps(convert(mes))

@app.route('/convert', methods=['POST'])
def post_mes():
  req = request.json['mes']
  res = convert(req)
  return json.dumps(res)

if __name__ == '__main__':
  app.run()
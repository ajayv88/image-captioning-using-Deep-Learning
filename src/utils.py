from keras.layers import *
import os
from keras.applications import resnet50, vgg16
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import numpy as np
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input
import tqdm
import string
from pickle import dump, load
import matplotlib.pyplot as plt


table = str.maketrans('', '', string.punctuation)
def build_model(model):
    print(model.summary())
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    return model

def findFeatures(model, path):
    dic = {}
    i = 0
    files = os.listdir(path)
    for file in tqdm.tqdm(files):
        img = image.load_img(path+file, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        features = model.predict(img)
        dic[file] = features
    return dic


def findImgs(path):
    f = open(path, 'r')
    dataset = []
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        dataset.append(line)
    
    return list(set(dataset))


def getDescriptions(path, Imgs):
    i = 0
    f = open(path, 'r')
    lines = f.readlines()
    dataset = []
    img_dataset = []
    dic = {}
    for line in lines:
        line = line.split('#')
        img, desc = line[0], line[1][1:]
        if img not in Imgs:
            continue
        dataset.append(desc)
        dic[desc] = img
        img_dataset.append(img)

    return dataset, dic, img_dataset




def trimWords(tweets):
    low_threshold = 1
    high_threshold = 100000000

    for t in range(0, len(tweets)):
        words = tweets[t].split()
        res = []
        for word in words:
            if len(word) == 1:
                pass
            else:
                res.append(word.lower())
        tweets[t] = ' '.join(res)

    return tweets

def trimPunctuations(tweets):
    for t in range(0, len(tweets)):
        words = tweets[t].split()
        res = []
        for word in words:
            word = removePunctuations(word)
            res.append(word.lower())
        tweets[t] = ' '.join(res)

    return tweets

def removePunctuations(tweet):
    global table
    tweet = tweet.translate(table)
    return tweet

def convertInput(tweets):
    vocab_to_int = {}
    int_to_vocab = {}
    max_len = -1
    num = 3
    final_res = []
    vocab_to_int["<start>"], int_to_vocab[1] = 1, "<start>"
    vocab_to_int["<end>"], int_to_vocab[2] = 2, "<end>"
    for t in range(0, len(tweets)):
        words = tweets[t].split()
        max_len = max(max_len, len(words)+2)
        res = []
        res.append("<start>")
        for word in words:
            word = removePunctuations(word)
            if word not in vocab_to_int:
                vocab_to_int[word] = num
                int_to_vocab[num] = word
                num += 1
            res.append(word.lower())
        res.append("<end>")
        final_res.append(res)

    return final_res, vocab_to_int, int_to_vocab, max_len

def make_numbers(descriptions, vocab_to_int, max_len):
    final_res = []
    for description in descriptions:
        res = []
        for word in description:
            res.append(vocab_to_int[word])
        final_res.append(res)
    return final_res


def prepareData(descriptions, img_dataset):
    descriptions = trimPunctuations(descriptions)
    descriptions = trimWords(descriptions)
    descriptions, vocab_to_int, int_to_vocab, max_len = convertInput(descriptions)
    descriptions = make_numbers(descriptions, vocab_to_int, max_len)
    descriptions = list(zip(img_dataset, descriptions))
    return descriptions, max_len, vocab_to_int, int_to_vocab, len(vocab_to_int) + 1


def makeinput(descriptions, max_len, corpus_size, feature_dic):
    img_features, rnn_input, rnn_output = [], [], []
    avg_len = 0
    for img, description in tqdm.tqdm(descriptions):
        for i in range(1, len(description)):
            input_desc, output_desc = description[:i], description[i]
            input_desc = input_desc + [0] * (max_len - len(input_desc))
            output_desc = to_categorical([output_desc], num_classes=corpus_size)[0]
            img_features.append(feature_dic[img])
            rnn_input.append(input_desc)
            rnn_output.append(output_desc)
    return img_features, rnn_input, rnn_output


def finalPrep(descriptions,  max_len, corpus_size, feature_dic):
    img_features, rnn_input, rnn_output = makeinput(descriptions, max_len, corpus_size, feature_dic)
    img_features, rnn_input, rnn_output = np.array(img_features), np.array(rnn_input), np.array(rnn_output)
    return img_features, rnn_input, rnn_output

def defineModel(corpus_size, max_len):
    #imageEncoder
    imageinput = Input(shape=(4096,))
    imageModel = Dropout(0.4)(imageinput)
    imageModel = Dense(256, activation="relu")(imageinput)
    
    #textEncoder
    textinput = Input(shape=(max_len,))
    textModel = Embedding(corpus_size, max_len, mask_zero=True)(textinput)
    textModel = Dropout(0.4)(textModel)
    textModel = LSTM(256)(textModel)
    
    #finalDecoder
    decoderModel = add([imageModel, textModel])
    decoderModel = Dense(256, activation="relu")(decoderModel)
    output = Dense(corpus_size, activation="softmax")(decoderModel)
    
    decoder = Model(inputs=[imageinput, textinput], outputs=output)
    decoder.compile(loss="categorical_crossentropy", optimizer="adam")
    return decoder


def makeTestSequence(sequence, vocab_to_int, max_len):
    res = []
    sequence = sequence.split()
    for word in sequence:
        res.append(vocab_to_int[word])
    res = res + [0] * (max_len - len(sequence))
    res = np.array(res)
    res = np.reshape(res, (1, max_len))
    return res

def testTimeFunctions(path, vggmodel, max_len, vocab_to_int, int_to_vocab, model):
    img = image.load_img(path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    features = vggmodel.predict(img)
    current_seq = "<start>"
    for i in range(max_len):
        rnn_input = makeTestSequence(current_seq, vocab_to_int, max_len)
        print(rnn_input.shape)
        rnn_output = model.predict([features, rnn_input])
        word = int_to_vocab[np.argmax(rnn_output)]
        current_seq += " " + word
        if word == "<end>":
            return current_seq

    return current_seq
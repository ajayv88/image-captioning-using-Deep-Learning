
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
from utils import *

DONE_ONCE = 1
TRAIN = 0
TEST = 1

table = str.maketrans('', '', string.punctuation)


def main():
    vgg16model = vgg16.VGG16(weights='imagenet')
    
    global DONE_ONCE
    dic = {}
    vgg16model = build_model(vgg16model)
    print(vgg16model.summary())

    if not DONE_ONCE:
        feature_dic = findFeatures(vgg16model, './../Flicker8k_Dataset/')
        dump(feature_dic, open('./../img_features.pkl', 'wb'))
    else:
        feature_dic = load(open('./../img_features.pkl', 'rb'))


    #READ IMAGES and ALL METADATA
    trainImgs = findImgs('./../Flickr8k_text/Flickr_8k.trainImages.txt')
    validImgs = findImgs('./../Flickr8k_text/Flickr_8k.devImages.txt')
    testImgs = findImgs('./../Flickr8k_text/Flickr_8k.testImages.txt')
    allImgs = list(set(trainImgs + validImgs + testImgs))
    all_size = len(allImgs)
    train_size = len(trainImgs)
    valid_size = len(validImgs)
    test_size = len(testImgs)
    print("Number of all Imgs {}".format(all_size))
    print("Number of training Imgs {}".format(train_size))
    print("Number of Validation Imgs {}".format(valid_size))
    print("Number of Test Imgs {}".format(test_size))


    #READ DESCRIPTIONS
    path = './../Flickr8k_text/Flickr8k.token.txt'
    all_descriptions, all_descriptions_dic, all_img_dataset = getDescriptions(path, allImgs)

    all_descriptions, max_len, vocab_to_int, int_to_vocab, corpus_size = prepareData(all_descriptions, all_img_dataset)
    train_descriptions = all_descriptions[:train_size*5]
    valid_descriptions = all_descriptions[train_size*5:(train_size+valid_size)*5]
    test_descriptions = all_descriptions[(train_size+valid_size)*5:]
    print("Maximum length of a description is {}".format(max_len))
    print("Vocabulary Size {}".format(corpus_size))


    train_img_features, train_rnn_input, train_rnn_output = finalPrep(train_descriptions, max_len, corpus_size, feature_dic)
    valid_img_features, valid_rnn_input, valid_rnn_output = finalPrep(valid_descriptions, max_len, corpus_size, feature_dic)
    test_img_features, test_rnn_input, test_rnn_output = finalPrep(test_descriptions, max_len, corpus_size, feature_dic)

    train_img_features = np.reshape(train_img_features, (train_img_features.shape[0], 4096))
    print("-----Training data shape------")
    print(train_img_features.shape, train_rnn_input.shape, train_rnn_output.shape)
    valid_img_features = np.reshape(valid_img_features, (valid_img_features.shape[0], 4096))
    print("-----Validation data shape------")
    print(valid_img_features.shape, valid_rnn_input.shape, valid_rnn_output.shape)
    test_img_features = np.reshape(test_img_features, (test_img_features.shape[0], 4096))
    print("-----Test data shape------")
    print(test_img_features.shape, test_rnn_input.shape, test_rnn_output.shape)

    if TRAIN:
        model = defineModel(corpus_size, max_len)
        filepath = 'bestmodel.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        model.fit([train_img_features, train_rnn_input], train_rnn_output, epochs=20, batch_size=64, verbose=2, callbacks=[checkpoint], validation_data=([valid_img_features, valid_rnn_input], valid_rnn_output))


    if TEST:
        # img_path = "./Flicker8k_Dataset/3100251515_c68027cc22.jpg"
        img_path = "./../test_imgs/test2.jpg"
        img = image.load_img(img_path, target_size=(224, 224))

        x = image.img_to_array(img)
        print(type(x))
        print(x.shape)
        plt.imshow(x/255.)
        plt.show()

        modelpath = "./bestmodel.h5"
        model = defineModel(corpus_size, max_len)
        model.load_weights(modelpath)
        print(type(model))
        sequence = testTimeFunctions(img_path, vgg16model, max_len, vocab_to_int, int_to_vocab, model)
        sequence = sequence.split()
        if sequence[0] == "<start>":
            print("yes")
            sequence = sequence[1:]
        if sequence[-1] == "<end>":
            print("yes")
            sequence = sequence[:-1]

        print("DESCRIPTION: " + ' '.join(sequence))

if __name__ == '__main__':
    main()




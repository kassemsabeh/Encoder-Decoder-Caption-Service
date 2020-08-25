import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pickle import load
from keras.models import Model, load_model
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.sequence import pad_sequences


def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
  return None

class EncoderDecoderModel():
    def __init__(self, tokenizer_path="tokenizer.p", model_path='models/model_9.h5'):
        self.__max_length = 32
        self.__tokenizer = load(open(tokenizer_path,"rb"))
        self.__model = load_model(model_path)
        self.__feature_extract_model = Xception(include_top=False, pooling="avg")
    
    def __evaluate_image(self, photo):
        in_text = 'start'
        for _ in range(self.__max_length):
            sequence = self.__tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.__max_length)
            pred = self.__model.predict([photo,sequence], verbose=0)
            pred = np.argmax(pred)
            word = word_for_id(pred, self.__tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        return in_text

    def __extract_features(self, filename):
        try:
            image = Image.open(filename)
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = self.__feature_extract_model.predict(image)
        return feature

    def predict_caption(self, img_path):
        photo = self.__extract_features(img_path)
        result = self.__evaluate_image(photo)
        caption = ' '.join(result.split(' ')[1:-1])
        return caption

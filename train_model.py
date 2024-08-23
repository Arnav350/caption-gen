import os
import numpy as np
import string
import pickle
import pandas as pd
from PIL import Image
from numpy import array
from tqdm import tqdm
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.models import Model

# Function to load and preprocess the captions
def load_descriptions(filename):
    df = pd.read_csv(filename)
    descriptions = {}
    for index, row in df.iterrows():
        image_id, caption = row['image'], row['caption']
        image_id = image_id.split('.')[0]
        if image_id not in descriptions:
            descriptions[image_id] = []
        descriptions[image_id].append(caption)
    return descriptions

# Clean the descriptions
def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)

# Convert the descriptions into a list of all descriptions
def to_lines(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# Fit a tokenizer given the descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# Calculate the length of the descriptions with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# Function to load an image and preprocess it for VGG16
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image

# Function to extract features from each photo in the directory
def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for name in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, name)
        image_id = name.split('.')[0]
        image = preprocess_image(filename)
        feature = model.predict(image, verbose=0)
        features[image_id] = feature
    return features

# Function to create sequences of input-output pairs
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = [], [], []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)

# Define the image captioning model
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Train the model
def train_model():
    # Load the descriptions
    descriptions = load_descriptions('archive/captions.txt')
    clean_descriptions(descriptions)
    
    # Prepare the tokenizer
    tokenizer = create_tokenizer(descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max_length(descriptions)
    
    # Save the tokenizer
    with open('backend/Models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Extract features from all images
    features = extract_features('archive/Images')
    
    # Create sequences of data for training
    X1, X2, y = create_sequences(tokenizer, max_len, descriptions, features, vocab_size)
    
    # Define the model
    model = define_model(vocab_size, max_len)
    
    # Fit the model
    model.fit([X1, X2], y, epochs=20, verbose=2)
    
    # Save the model
    model.save('backend/Models/caption_model.h5')

if __name__ == "__main__":
    train_model()

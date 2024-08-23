from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import pickle

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and tokenizer
model = load_model('backend/Models/caption_model.h5')
with open('backend/Models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 34  # Set this based on your training

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image

def predict_caption(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file.stream)
    image = preprocess_image(image)
    
    # Extract features from image
    feature = model.predict(image, verbose=0)
    
    # Generate caption
    caption = predict_caption(feature)
    
    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(debug=True)

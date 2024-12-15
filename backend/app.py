from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the trained model
model = load_model('hate_speech_model.h5')  # Update with the actual path to your model file

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:  # Update with the actual path to your tokenizer file
    tokenizer = pickle.load(handle)

# Set maxlen (ensure this matches the value used during training)
maxlen = 50

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the form."""
    try:
        # Get text input from the form
        text = request.form.get("text")
        if not text:
            return render_template("index.html", result="Please enter some text to analyze.")

        # Preprocess the input text
        sequences = tokenizer.texts_to_sequences([text])
        
        # Debugging: Print the length of the sequence
        print(f"Length of tokenized sequence: {len(sequences[0])}")
        
        # Pad the sequence
        padded_seq = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

        # Debugging: Print the shape of padded_seq
        print(f"Shape of padded_seq: {padded_seq.shape}")

        # Ensure the input shape matches the model's expected shape
        if padded_seq.shape[1] != maxlen:
            return render_template("index.html", result=f"Error: Input sequence length is incorrect. Expected {maxlen}, but got {padded_seq.shape[1]}.")

        # If the padded sequence length is greater than maxlen, truncate it manually
        if padded_seq.shape[1] > maxlen:
            padded_seq = padded_seq[:, :maxlen]
        
        # Predict using the model
        prediction = model.predict(padded_seq)
        is_hate_speech = bool(prediction[0][0] > 0.5)  # Assuming binary classification

        # Generate the result
        result = "Hate Speech Detected" if is_hate_speech else "No Hate Speech Detected"
        return render_template("index.html", result=result, input_text=text)
    
    except Exception as e:
        # Handle any errors and display them on the page
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

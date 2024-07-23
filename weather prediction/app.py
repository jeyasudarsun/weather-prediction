from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Path for saving uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model_path = r"C:\Users\pravi\OneDrive\Desktop\weather prediction\weather prediction project\model.h5"
model = load_model(model_path)

# Define the labels
labels = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    X = X / 255.0
    preds = model.predict(X)
    pred = np.argmax(preds, axis=1)[0]
    result = str(labels[pred])
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']

        # Save the file to ./uploads
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)

        # Redirect to result page
        return redirect(url_for('result', prediction=result, img_path=file_path))

    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    img_path = request.args.get('img_path')
    return render_template('result.html', prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(os.getcwd(), filename))
        return redirect(url_for('uploaded'))

@app.route('/uploads/<filename>')
def uploaded(filename):
    return send_from_directory(os.getcwd(), filename)

if __name__ == '__main__':
    app.run(debug=True)

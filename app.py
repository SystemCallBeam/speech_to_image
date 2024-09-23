
from flask import Flask, render_template, request, jsonify, send_file
from controller import start_recording, stop_recording, get_past_entries

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    start_recording()
    return jsonify({"status": "recording started"})

@app.route('/stop', methods=['POST'])
def stop():
    text, image_path = stop_recording()
    return jsonify({"status": "recording stopped", "text": text, "image_path": image_path})

@app.route('/past', methods=['GET'])
def past_entries():
    entries = get_past_entries()
    return jsonify(entries)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import subprocess
import threading

def run(instruction, callback_url):
    prompt = f"Send a POST HTTP request to {callback_url}. The body of the request should be a JSON object with two keys: 1. 'status': The value of this key should be the integer 200. 2. 'response': The value of this key should be the answer in text to the following instruction: '{instruction}'. If an error occurs, the 'status' key should have the value 500 and the 'response' should have the error with the backtrace. Send the request only once and terminate immediately after sending the request."
    command = f"echo '{prompt}' | python -m autogpt --continuous --continuous-limit 100 --allow-downloads --skip-news --skip-reprompt --browser-name firefox"
    subprocess.run(command, shell=True)

app = Flask(__name__)

# Define a POST endpoint to execute shell commands
@app.route('/execute', methods=['POST'])
def execute_command():
    payload = request.get_json()
    instruction = payload.get('instruction')
    callback_url = payload.get('callback_url')
    try:
        thread = threading.Thread(target=run, args=[instruction, callback_url])
        thread.start()
        return jsonify({'response': 'acknowledged'})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e.output.decode('utf-8'))}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0')
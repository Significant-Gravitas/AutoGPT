from flask import Flask, request, jsonify
from autogpt.agent.agent_manager import AgentManager
from autogpt.commands.file_operations import write_to_file, read_file

app = Flask(__name__)
agent_manager = AgentManager()

@app.route('/agents', methods=['POST'])
def create_agent():
    data = request.json
    task = data.get('task')
    prompt = data.get('prompt')
    model = data.get('model', 'gpt-3.5-turbo')
    key, response = agent_manager.create_agent(task, prompt, model)
    return jsonify({'key': key, 'response': response})

@app.route('/agents', methods=['GET'])
def list_agents():
    agents = agent_manager.list_agents()
    return jsonify(agents)

@app.route('/agents/<int:key>', methods=['DELETE'])
def delete_agent(key):
    success = agent_manager.delete_agent(key)
    return jsonify({'success': success})

@app.route('/save', methods=['POST'])
def save_information():
    data = request.json
    filename = data.get('filename')
    content = data.get('content')
    response = write_to_file(filename, content)
    return jsonify({'response': response})

@app.route('/retrieve', methods=['GET'])
def retrieve_information():
    filename = request.args.get('filename')
    content = read_file(filename)
    return jsonify({'content': content})

if __name__ == '__main__':
    app.run(debug=True)

# Running Ollama with AutoGPT

Follow these steps to set up and run Ollama and your AutoGPT project:

1. **Run Ollama**
   - Open a terminal
   - Execute the following command:
     ```
     ollama run llama3
     ```
   - Leave this terminal running

2. **Run the Backend**
   - Open a new terminal
   - Navigate to the backend directory in the AutoGPT project:
     ```
     cd rnd/autogpt_server/
     ```
   - Start the backend using Poetry:
     ```
     poetry run app
     ```

3. **Run the Frontend**
   - Open another terminal
   - Navigate to the frontend directory in the AutoGPT project:
     ```
     cd rnd/autogpt_builder/
     ```
   - Start the frontend development server:
     ```
     npm run dev
     ```

4. **Choose the Ollama Model**
   - Add LLMBlock in the UI
   -  Choose the last option in the model selection dropdown
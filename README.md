# SoloAgent: AI-Powered MVP Generator

## Overview
SoloAgent is an innovative AI-powered tool designed to streamline the development of Minimum Viable Products (MVPs). By leveraging AI, SoloAgent allows developers to input prompts and receive both backend and frontend code as outputs, including unit tests, drastically reducing the time and effort required to build and test a fully functional MVP across various platforms and technologies.

## Features
- **AI-Powered Code Generation:** Input a prompt and receive backend and frontend code.
- **Unit Test Generation:** Automatically generate unit tests for the generated code.
- **Platform Agnostic:** Supports multiple platforms and technologies for versatile MVP development.
- **End-to-End Solution:** Provides a complete solution, from backend logic to frontend interface and testing.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Node.js and npm
- An OpenAI API key

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/botirk38/SoloAgent.git
   cd SoloAgent
   ```

2. **Set up a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ./run setup
   poetry shell
    
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

### Usage
```
  ./run start SoloAgent
```

### Example Prompt
```
Create a simple to-do application with user authentication. The backend should be built with Node.js and Express, and the frontend should use React. Include unit tests for the API endpoints and React components.
```

## Documentation
Detailed documentation explaining the approach, the AI model used, how the prompts are processed, and the code generation process can be found in the [docs](docs) directory.

## Contributing
We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions or feedback, please open an issue on GitHub or contact us at [btrghstk@gmail.com].

---

Get ready to revolutionize MVP development with SoloAgent. We look forward to your feedback and contributions!

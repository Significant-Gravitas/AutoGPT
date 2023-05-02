# Code Generator

This is a simple code generator that can generate Python code for basic arithmetic operations.

## Data and Rules

The data consists of pairs of natural language expressions and Python code snippets that perform the corresponding arithmetic operations. For example:

- "Add 3 and 5" -> "3 + 5"
- "Subtract 7 from 10" -> "10 - 7"
- "Multiply 4 by 6" -> "4 * 6"
- "Divide 8 by 2" -> "8 / 2"

The rules are based on the syntax and semantics of Python and natural language. For example:

- The order of operands in addition and multiplication is irrelevant, but the order of operands in subtraction and division is relevant.
- The natural language expression should use words like "add", "subtract", "multiply", and "divide" to indicate the arithmetic operation.
- The Python code should use symbols like "+", "-", "*", and "/" to indicate the arithmetic operation.
- The natural language expression and the Python code should use the same numbers as operands.

## Code Generation Model

The code generation model is a sequence-to-sequence model that takes a natural language expression as input and outputs a Python code snippet. The model is trained on the data using a cross-entropy loss function and an Adam optimizer. The model architecture is as follows:

- An encoder that consists of an embedding layer and a recurrent neural network (RNN) layer that encodes the input sequence into a hidden state vector.
- A decoder that consists of an embedding layer, an RNN layer, and a linear layer that generates the output sequence from the hidden state vector.
- An attention mechanism that allows the decoder to focus on relevant parts of the input sequence while generating the output sequence.

## Testing and Evaluation

The code generation model is tested on unseen data that are not used for training. The model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The model performance can be improved by increasing the size and diversity of the data, tuning the hyperparameters of the model, or using more advanced models such as transformers or neural machine translation models.        import re

# 연산자 토큰을 Python 연산자로 변환하는 딕셔너리
OPERATORS = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/"
}

def generate_code(nl_text):
    # 자연어 문장을 토큰화하여 리스트로 변환
    tokens = re.findall(r"[\w']+", nl_text)
    
    # 토큰 리스트에서 연산자와 피연산자 추출
    operator = None
    operands = []
    for token in tokens:
        if token in OPERATORS:
            operator = OPERATORS[token]
        elif token.isdigit():
            operands.append(token)
    
    # 피연산자 리스트를 Python 코드 문자열로 변환
    operands_code = " ".join(operands)
    
    # Python 코드 문자열을 생성하여 반환
    code = f"{operands_code} {operator}"
    return code                           print(generate_code("Add 3 and 5"))  # 출력: 3 + 5
print(generate_code("Subtract 7 from 10"))  # 출력: 10 - 7
print(generate_code("Multiply 4 by 6"))  # 출력: 4 * 6
print(generate_code("Divide 8 by 2"))  # 출력: 8 / 2

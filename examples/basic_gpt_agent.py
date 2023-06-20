import json
import openai
from agbenchmark.benchmark.challenges.retrieval.r1_test import RetrievelChallenge


def basic_gpt_agent(challenge_file):
    challenge = RetrievelChallenge.from_json_file(challenge_file)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": challenge.agent_input}])
    answer = response["choices"][0]["message"]["content"]

    output_file = "./basic_gpt_agent_retrieval_results.txt"
    with open(output_file, "w") as f:
        f.write(answer)

    print("QUERY       : ", challenge.agent_input)
    print("AGENT ANSWER: ", answer)

    score = challenge.run(output_file)

    print("AGENT SCORE : ", score)

if __name__ == "__main__":
    basic_gpt_agent("./data/retrieval/r1_test_data_1.json")

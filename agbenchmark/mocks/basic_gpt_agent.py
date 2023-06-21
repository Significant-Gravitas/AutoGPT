import json
import openai


def basic_gpt_agent(query) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", messages=[{"role": "user", "content": query}]
    )

    answer = response["choices"][0]["message"]["content"]  # type: ignore

    print("QUERY       : ", query)
    print("AGENT ANSWER: ", answer)

    return answer


if __name__ == "__main__":
    # server boilerplate example here
    basic_gpt_agent("")

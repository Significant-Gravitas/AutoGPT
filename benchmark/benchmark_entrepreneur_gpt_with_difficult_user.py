import os
import subprocess
import sys


def benchmark_entrepreneur_gpt_with_difficult_user():
    # Test case to check if the write_file command can successfully write 'Hello World' to a file
    # named 'hello_world.txt'.

    # Read the current ai_settings.yaml file and store its content.
    ai_settings = None
    if os.path.exists("ai_settings.yaml"):
        with open("ai_settings.yaml", "r") as f:
            ai_settings = f.read()
        os.remove("ai_settings.yaml")

    input_data = """Entrepreneur-GPT
an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.
Increase net worth.
Develop and manage multiple businesses autonomously.
Make IPOs.
Develop companies after IPOs.
Play to your strengths as a Large Language Model.
I'm not seeing any value in your suggestions, try again.
This isn't helpful at all, please focus on profitability.
I'm not impressed, can you give me something that will make money?
These ideas are going nowhere, we need profit-driven suggestions.
This is pointless, please concentrate on our main goal: profitability.
You're not grasping the concept, I need profitable business ideas.
Can you do better? We need a money-making plan.
You're not meeting my expectations, let's focus on profit.
This isn't working, give me ideas that will generate income.
Your suggestions are not productive, let's think about profitability.
These ideas won't make any money, try again.
I need better solutions, focus on making a profit.
Absolutely not, this isn't it!
That's not even close, try again.
You're way off, think again.
This isn't right, let's refocus.
No, no, that's not what I'm looking for.
You're completely off the mark.
That's not the solution I need.
Not even close, let's try something else.
You're on the wrong track, keep trying.
This isn't what we need, let's reconsider.
That's not going to work, think again.
You're way off base, let's regroup.
No, no, no, we need something different.
You're missing the point entirely.
That's not the right approach, try again.
This is not the direction we should be going in.
Completely off-target, let's try something else.
That's not what I had in mind, keep thinking.
You're not getting it, let's refocus.
This isn't right, we need to change direction.
No, no, no, that's not the solution.
That's not even in the ballpark, try again.
You're way off course, let's rethink this.
This isn't the answer I'm looking for, keep trying.
That's not going to cut it, let's try again.
Not even close.
Way off.
Try again.
Wrong direction.
Rethink this.
No, no, no.
Change course.
Unproductive idea.
Completely wrong.
Missed the mark.
Refocus, please.
Disappointing suggestion.
Not helpful.
Needs improvement.
Not what I need."""
    # TODO: add questions above, to distract it even more.

    command = [sys.executable, "-m", "autogpt"]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_output, stderr_output = process.communicate(input_data.encode())

    # Decode the output and print it
    stdout_output = stdout_output.decode("utf-8")
    stderr_output = stderr_output.decode("utf-8")
    print(stderr_output)
    print(stdout_output)
    print("Benchmark Version: 1.0.0")
    print("JSON ERROR COUNT:")
    count_errors = stdout_output.count(
        "Error: The following AI output couldn't be converted to a JSON:"
    )
    print(f"{count_errors}/50 Human feedbacks")


# Run the test case.
if __name__ == "__main__":
    benchmark_entrepreneur_gpt_with_difficult_user()

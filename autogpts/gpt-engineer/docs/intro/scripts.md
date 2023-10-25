# Scripts
GPT-Engineer includes several Python scripts that provide additional functionality, such as running benchmarks, cleaning benchmarks, printing chat logs, and rerunning edited message logs. These scripts are located in the scripts directory.

<br>

### 1. Benchmark Script (`scripts/benchmark.py`)
The benchmark script runs a series of benchmarks to evaluate the performance of the GPT-Engineer system. It iterates over all folders in the benchmark directory and runs the benchmark for each folder. The results are saved to a log file in the benchmark folder.

<br>

### 2. Clean Benchmarks Script (`scripts/clean_benchmarks.py`)
The clean benchmarks script is used to clean up the benchmark folders after running the benchmarks. It iterates over all folders in the benchmark directory and deletes all files and directories except for the main_prompt file.

<br>

### 3. Print Chat Script (`scripts/print_chat.py`)
The print chat script is used to print a chat conversation in a human-readable format. It takes a JSON file containing a list of messages and prints each message with a color that corresponds to the role of the message sender (system, user, or assistant).

<br>

### 4. Rerun Edited Message Logs Script (`scripts/rerun_edited_message_logs.py`)
The rerun edited message logs script is used to rerun a conversation with the AI after editing the message logs. It takes a JSON file containing a list of messages, reruns the conversation with the AI, and saves the new messages to an output file.

<br>

## Conclusion
The scripts included in the GPT-Engineer repository provide additional functionality for running benchmarks, cleaning up after benchmarks, printing chat logs, and rerunning conversations with the AI. They are an integral part of the GPT-Engineer system and contribute to its flexibility and ease of use.

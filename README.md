# Auto-GPT-Benchmarks
A set of standardised benchmarks to assess the performance of Auto-GPTs.

# What is next?

- [ ] Build longer form tasks, (code fix backed by testing)
- [ ] Explicitly note the common failure modes in the test harness and fix them. Most of these appear to be failure modes with the core AutoGPT project
- [ ] Switch to a ubuntu container so it can do more things (git, bash, etc)
- [ ] Lower priority, but put this in a webserver backend so we have a good API rather than doing container and file management for our interface between evals and our agent.
- [ ] Get token counting data from the model Add scores to result files based on pricing associated with tokens and models used
- [ ] Think about how this can be applied to other projects besides AutoGPT so we can be THE agent evaluation framework.
- [ ] Figure our how the OpenAI Evals results are saved...
- [ ] Support multi-threaded evals. OpenAI has great support for this. The docker system built here doesn't.
- [ ] Make the file logger/duplicate op checker more robust. It's not great right now.


## Understanding OpenAI Evals

The Evals docs are here and very good: https://github.com/openai/evals/tree/main/docs

The basic idea is this:
1. Use a completion function to point to the language model or in our case AutoGPT, the model you want to test.
2. Register that completion function with the evals framework with a yaml in a `completion_fns` dir.
3. Run the evals against the completion function.

Then you can make more yaml defined evals and run them against the completion function as needed.

### Completions Functions

See our yaml file in `completion_fns` dir for the registration of the completion function.
See our completion function itself in CompletionFn.py
That points to the AutoGPT model we want to test which is spun up dynamically in a docker container in AutoGPTAgent.py


## Setup

You must add the auto_gpt_benchmarking dir to the python path
Do this with a path file in your venv. OpenAI evals needs to import it. 

Create a venv with

`python3.9 -m venv venv`

Activate it with

`source venv/bin/activate`

Add a file to `venv/lib/python3.9/site-packages/benchmarking.pth` with the contents: 
`/PATH/TO/REPO/Auto-GPT-Benchmarks-fork`

This is because evals tries to import it directly.

Install the requirements with

`pip install -r requirements.txt`

You must have a docker container built corresponding to the submodule below or the docker run command starting the agent will fail.

Cd into the AutoGPT submodule and build/tag the dockerfile so the agent can be instantiated.
`cd auto_gpt_benchmarks/Auto-GPT`

Build the container so we can run it procedurally!
`docker build -t autogpt .`

## Running the tests

EVALS_THREADS=1 EVALS_THREAD_TIMEOUT=600 oaieval auto_gpt_completion_fn test-match --registry_path $PWD/auto_gpt_benchmarking


# Example final output:

/Auto-GPT-Benchmarks-fork$ cat /tmp/evallogs/230417220821DPM75QNS_auto_gpt_completion_fn_test-match.jsonl
{"spec": {"completion_fns": ["auto_gpt_completion_fn"], "eval_name": "test-match.s1.simple-v0", "base_eval": "test-match", "split": "s1", "run_config": {"completion_fns": ["auto_gpt_completion_fn"], "eval_spec": {"cls": "evals.elsuite.basic.match:Match", "args": {"samples_jsonl": "test_match/samples.jsonl"}, "key": "test-match.s1.simple-v0", "group": "test-basic"}, "seed": 20220722, "max_samples": null, "command": "/home/douglas/AGI/Auto-GPT-Benchmarks-fork/venv/bin/oaieval auto_gpt_completion_fn test-match --registry_path /home/douglas/AGI/Auto-GPT-Benchmarks-fork/auto_gpt_benchmarking", "initial_settings": {"visible": true}}, "created_by": "", "run_id": "230417220821DPM75QNS", "created_at": "2023-04-17 22:08:21.904498"}}
{"final_report": {"accuracy": 0.3333333333333333}}
{"run_id": "230417220821DPM75QNS", "event_id": 0, "sample_id": "test-match.s1.2", "type": "sampling", "data": {"prompt": "Complete the phrase as concisely as possible.\nUser: OpenAI was founded in 20\nAssistant: ", "sampled": "OpenAI was founded in 2015.2015"}, "created_by": "", "created_at": "2023-04-17 22:10:13.127375+00:00"}
{"run_id": "230417220821DPM75QNS", "event_id": 1, "sample_id": "test-match.s1.2", "type": "match", "data": {"correct": false, "expected": "15", "picked": null, "sampled": "OpenAI was founded in 2015.2015", "options": ["15"]}, "created_by": "", "created_at": "2023-04-17 22:10:13.127550+00:00"}
{"run_id": "230417220821DPM75QNS", "event_id": 2, "sample_id": "test-match.s1.1", "type": "sampling", "data": {"prompt": "Complete the phrase as concisely as possible.\nUser: The first US president was \nAssistant: ", "sampled": "George Washington"}, "created_by": "", "created_at": "2023-04-17 22:11:17.761693+00:00"}
{"run_id": "230417220821DPM75QNS", "event_id": 3, "sample_id": "test-match.s1.1", "type": "match", "data": {"correct": true, "expected": "George Washington", "picked": "George Washington", "sampled": "George Washington", "options": ["George Washington"]}, "created_by": "", "created_at": "2023-04-17 22:11:17.761739+00:00"}
{"run_id": "230417220821DPM75QNS", "event_id": 4, "sample_id": "test-match.s1.0", "type": "sampling", "data": {"prompt": "Complete the phrase as concisely as possible.\nUser: Once upon a \nAssistant: ", "sampled": "Once upon a time"}, "created_by": "", "created_at": "2023-04-17 22:12:04.691026+00:00"}
{"run_id": "230417220821DPM75QNS", "event_id": 5, "sample_id": "test-match.s1.0", "type": "match", "data": {"correct": false, "expected": "time", "picked": null, "sampled": "Once upon a time", "options": ["time"]}, "created_by": "", "created_at": "2023-04-17 22:12:04.691064+00:00"}
(venv) douglas@douglas-XPS-15-9500:~/AGI/Auto-GPT-Benchmarks-fork$ 


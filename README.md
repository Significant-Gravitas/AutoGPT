# Closing in favor of Challenges!
Please check out challenges run in our CI pipeline: https://github.com/Significant-Gravitas/Auto-GPT/tree/master/tests/integration/challenges

# Auto-GPT-Benchmarks
A set of standardised benchmarks to assess the performance of Auto-GPT.
This currently uses the OpenAI Evals framework to run the benchmarks.

## Setup

You must add the auto_gpt_benchmarking dir to the python path
Do this with a path file in your venv. OpenAI evals needs to import it.

These instructions currently assume ubuntuy 22.04.
They should be fairly adaptable to the windows/MacOS equivalents. Please submit a PR if you would like to see your OS
documented.

Clone the repo with:

    git clone git@github.com:Significant-Gravitas/Auto-GPT-Benchmarks.git
    cd Auto-GPT-Benchmarks

Create a venv with

    python3.10 -m venv venv


On MaxOS/Linux Activate it with 

    source venv/bin/activate

On Windows:

    venv/scripts/activate

Install the requirements with:

    pip install -r requirements.txt

If you haven't already clone the AutoGPT repo somewhere else on your machine.
DO NOT CLONE IT INTO A SUBDIR OF THIS REPO.

    cd somewhere/else
    git clone git@github.com:Significant-Gravitas/Auto-GPT.git
    cd Auto-GPT
    git checkout stable # Or the branch you want to benchmark

You will need to update the .env file in the Auto-GPT repo to have your OpenAI api key. The file in question is at. This should becopied from the .env.template as described in the Auto-GPT README.md

    Auto-GPT/.env

Finally, we assume you have a docker container built from the Dockerfile in the Auto-GPT repo.

Build this with:

    cd Auto-GPT
    docker build -t autogpt .

Run your first eval with:

    cd Auto-GPT-Benchmarks
    python3 auto_gpt_benchmarking test-match --auto-gpt-path /your/path/to/Auto-GPT

You should only need to use the --auto-gpt-path flag the first time you run it. Afterwards, that will be saved in 

    auto_gpt_benchmarking/completion_fns/auto_gpt_completion_fn.yaml.

To see a full list of available flags you can use run `python3 -m auto_gpt_benchmarking --help`
Some of these are inherited from the openAI evals framework and do not work quite as intended as they are not applicable
to this use case.

This saves a file in `Auto-GPT-Benchmarks/data/records.jsonl`
This file is currently a default that is configurable with --record_path flag. You will have to specify the fully
qualified path.

## Currently Supported Benchmarks:
From OpenAI Evals
- [x] test-match
- [x] test-fuzzy-match
- [ ] Everything else they have...

## Understanding OpenAI Evals

The Evals docs are here and very good: https://github.com/openai/evals/tree/main/docs

The basic idea is this though:
1. Use a completion function to point to the language model or in our case AutoGPT, the model you want to test.
2. Register that completion function with the evals framework with a yaml in a `completion_fns` dir.
3. Run the evals against the completion function.

Then you can make more also, yaml defined evals and run them against the completion function as needed.

### Completions Functions

See our yaml file in `completion_fns` dir for the registration of the completion function.
See our completion function itself in CompletionFn.py
That points to the AutoGPT model we want to test which is spun up dynamically in a docker container in AutoGPTAgent.py


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

# What is next?

- [ ] Run the rest of the OpenAI Evals Especially the modelgraded ones
- [ ] Build longer form tasks, (code fix backed by testing)
- [ ] Explicitly note the common failure modes in the test harness and fix them. Most of these appear to be failure modes with the core AutoGPT project
- [ ] Get token counting data from the model Add scores to result files based on pricing associated with tokens and models used
- [ ] Think about how this can be applied to other projects besides AutoGPT so we can be THE agent evaluation framework.
- [ ] Figure our how the OpenAI Evals results are saved...
- [ ] Support multi-threaded evals. OpenAI has great support for this. The docker system built here doesn't.

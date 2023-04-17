# Auto-GPT-Benchmarks
A set of standardised benchmarks to assess the performance of Auto-GPTs.

# What is next?

- [ ] Build longer form tasks, (code fix backed by testing)
- [ ] Explicitly note the common failure modes in the test harness and fix them. Most of these appear to be failure modes with the core AutoGPT project
- [ ] Switch to a ubuntu container so it can do more things (git, bash, etc)
- [ ] Lower priority, but put this in a webserver backend so we have a good API
- [ ] Get token counting data from the model Add scores to result files based on pricing associated with tokens and models used
- [ ] Think about how this can be applied to other projects besides AutoGPT so we can be THE agent evaluation framework.
- [ ] Figure our how the OpenAI Evals results are saved...


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


# RANDOM SHIT

You must add the auto_gpt_bencchmarking dir to the python path
Do this with a path file in your venv. OpenAI evals needs to import it. 

I added a file to `venv/lib/python3.9/site-packages/benchmarking.pth` with the contents: 
`/home/douglas/AGI/Auto-GPT-Benchmarks-fork`



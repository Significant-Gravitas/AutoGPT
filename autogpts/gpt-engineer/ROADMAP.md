# Roadmap

<img width="800" alt="image" src="https://github.com/AntonOsika/gpt-engineer/assets/4467025/1af9a457-1ed8-4ab3-90e6-4fa52bbb8c09">


There are three main milestones we believe will greatly increase gpt-engineer's reliability and capability:
- [x] Continuous evaluation of our progress 🎉
- [ ] Test code and fix errors with LLMs
- [ ] Make code generation become small, verifiable steps

## Our current focus:

- [x] **Continuous evaluation of progress 🎉**
  - [x] Create a step that asks “did it run/work/perfect” in the end of each run [#240](https://github.com/AntonOsika/gpt-engineer/issues/240) 🎉
  - [x] Collect a dataset for gpt engineer to learn from, by storing code generation runs 🎉
  - [ ] Run the benchmark multiple times, and document the results for the different "step configs" [#239](https://github.com/AntonOsika/gpt-engineer/issues/239)
  - [ ] Improve the default config based on results
- [ ] **Self healing code**
  - [ ] Run the generated tests
  - [ ] Feed the results of failing tests back into LLM and ask it to fix the code
- [ ] **Let human give feedback**
  - [ ] Ask human for what is not working as expected in a loop, and feed it into LLM to fix the code, until the human is happy
- [ ] **Improve existing projects**
  - [ ] Decide on the "flow" for the CLI commands and where the project files are created
  - [ ] Add an "improve code" command
  - [ ] Benchmark capabilities against other tools

## Experimental research
This is not our current focus, but if you are interested in experimenting: Please
create a thread in Discord #general and share your intentions and your findings as you
go along. High impact examples:
- [ ] **Make code generation become small, verifiable steps**
  - [ ] Ask GPT4 to decide how to sequence the entire generation, and do one
  prompt for each subcomponent
  - [ ] For each small part, generate tests for that subpart, and do the loop of running the tests for each part, feeding
results into GPT4, and let it edit the code until they pass
- [ ] **Ad hoc experiments**
  - [ ] Try Microsoft guidance, and benchmark if this helps improve performance
  - [ ] Dynamic planning: Let gpt-engineer plan which "steps" to carry out itself, depending on the
task, by giving it few shot example of what are usually "the right-sized steps" to carry
out for such projects

## Codebase improvements
By improving the codebase and developer ergonomics, we accelerate progress. Some examples:
- [ ] Set up automatic PR review for all PRs with e.g. Codium pr-agent
- [ ] LLM tests in CI: Run super small tests with GPT3.5 in CI, that check that simple code generation still works

# How you can help out

You can:

- Post a "design" as a google doc in our Discord and ask for feedback to address one of the items in the roadmap
- Submit PRs to address one of the items in the roadmap
- Do a review of someone else's PR and propose next steps (further review, merge, close)

Volunteer work in any of these will get acknowledged.

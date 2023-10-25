=====
Usage
=====

Setup
=====

With an OpenAI API key (preferably with GPT-4 access) run:

- `export OPENAI_API_KEY=[your api key]`

To set API key on windows check the [Windows README](.github/WINDOWS_README.md).

Run
===

- Create an empty folder. If inside the repo, you can run:
  - `cp -r projects/example/ projects/my-new-project`
- Fill in the `prompt` file in your new folder
- `gpt-engineer projects/my-new-project`
  - (Note, `gpt-engineer --help` lets you see all available options. For example `--steps use_feedback` lets you improve/fix code in a project)

By running gpt-engineer you agree to our [terms](https://github.com/AntonOsika/gpt-engineer/blob/main/TERMS_OF_USE.md).

Results
=======
- Check the generated files in `projects/my-new-project/workspace`


To **run in the browser** you can simply:

.. image:: https://github.com/codespaces/badge.svg
   :target: https://github.com/AntonOsika/gpt-engineer/codespaces

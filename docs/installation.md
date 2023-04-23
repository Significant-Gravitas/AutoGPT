# 💾 Installation

## Prerequisites
- [Git](https://git-scm.com/)
- [OpenAI API key](https://platform.openai.com/account/api-keys)
- [Paid OpenAI Account](https://platform.openai.com/account/billing/overview)

**Important!** You **must** have a paid account via ![OpenAI API > Billing](./docs/imgs/openai-api-key-billing-paid-account.png). We also highly recommend tracking your usage on the [Usage](https://platform.openai.com/account/usage) page, and setting a spending limit on the [Usage limits](https://platform.openai.com/account/billing/limits) page.

## Steps
1. Clone this repository:
``` shell
    git clone -b stable https://github.com/Significant-Gravitas/Auto-GPT.git
```

2. Navigate to the project root:
``` shell
    cd Auto-GPT
```

3. Install the required dependencies:

``` shell
    pip install -r requirements.txt
```

4. Copy the `.env.template` file into a file named `.env`: 
```bash
cp .env.template .env
```

5. Add your `OPENAI_API_KEY` to `.env`.

### Docker

You can also build and run this project in a Docker image.

1. To build and run this project, run:
``` shell
docker build -t autogpt .
docker run -it --env-file=./.env -v $PWD/auto_gpt_workspace:/home/appuser/auto_gpt_workspace autogpt
```

2. Optionally, if you have `docker-compose`, run:
``` shell
docker-compose run --build --rm auto-gpt
```

3. To pass extra arguments, see the following examples to run with `--gpt3only` and `--continuous` mode:
``` shell
docker run -it --env-file=./.env -v $PWD/auto_gpt_workspace:/home/appuser/auto_gpt_workspace autogpt --gpt3only --continuous
```

``` shell
docker-compose run --build --rm auto-gpt --gpt3only --continuous
```

Alternatively, you can pull the latest release directly from [Docker Hub](https://hub.docker.com/r/significantgravitas/auto-gpt).


### Azure (optional)
If you want to use GPT on an Azure instance, set `USE_AZURE` to `True` and follow these instructions.

1. Copy the `azure.yaml.template` file into a file named `azure.yaml`: 
```bash
cp azure.yaml.template azure.yaml
```
2. Add the following credentials to the `azure_model_map` section:
   1. `fast_llm_model_deployment_id`: "`Your gpt-3.5-turbo or gpt-4 deployment ID`"
   2. `smart_llm_model_deployment_id`: "`Your gpt-4 deployment ID`"
   3. `embedding_model_deployment_id`: "`Your text-embedding-ada-002 v2 deployment ID`"
   4. To learn more, see the **Microsoft Azure Endpoints** section in the [OpenAI Python Library](https://pypi.org/project/openai/), and [Tutorial: Explore Azure OpenAI Service embeddings and document search](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=command-line) for the embedding model.
3. If you're on Windows you may need to install [Microsoft Visual C++ Redistributable latest supported downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

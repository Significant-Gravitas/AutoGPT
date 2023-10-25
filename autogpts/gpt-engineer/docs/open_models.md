Using with open/local models
============================

You can integrate `gpt-engineer` with open-source models by leveraging an OpenAI-compatible API. One such API is provided by the [text-generator-ui _extension_ openai](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/openai/README.md).

Setup
-----

To get started, first set up the API with the Runpod template, as per the [instructions](https://github.com/oobabooga/text-generation-webui/blob/main/extensions/openai/README.md>).

Running the Example
-------------------

Once the API is set up, you can find the host and the exposed TCP port by checking your Runpod dashboard.

Then, you can use the port and host to run the following example using WizardCoder-Python-34B hosted on Runpod:

```
  OPENAI_API_BASE=http://<host>:<port>/v1 python -m gpt_engineer.cli.main benchmark/pomodoro_timer --steps benchmark TheBloke_WizardCoder-Python-34B-V1.0-GPTQ
```

Using Azure models
==================

You set your Azure OpenAI key:
- `export OPENAI_API_KEY=[your api key]`

Then you call `gpt-engineer` with your service endpoint `--azure https://aoi-resource-name.openai.azure.com` and set your deployment name (which you created in the Azure AI Studio) as the model name (last `gpt-engineer` argument).

Example:
`gpt-engineer --azure https://myairesource.openai.azure.com ./projects/example/ my-gpt4-project-name`

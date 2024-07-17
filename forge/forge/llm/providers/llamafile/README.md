# Llamafile Integration Notes

Tested with:
* Python 3.11
* Apple M2 Pro (32 GB), macOS 14.2.1
* quantized mistral-7b-instruct-v0.2

## Setup

Download a `mistral-7b-instruct-v0.2` llamafile:
```shell
wget -nc https://huggingface.co/jartine/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.llamafile
chmod +x mistral-7b-instruct-v0.2.Q5_K_M.llamafile
./mistral-7b-instruct-v0.2.Q5_K_M.llamafile --version
```

Run the llamafile server:
```shell
LLAMAFILE="./mistral-7b-instruct-v0.2.Q5_K_M.llamafile"

"${LLAMAFILE}" \
--server \
--nobrowser \
--ctx-size 0 \
--n-predict 1024

# note: ctx-size=0 means the prompt context size will be set directly from the
# underlying model configuration. This may cause slow response times or consume
# a lot of memory.
```

## TODOs

* `SMART_LLM`/`FAST_LLM` configuration: Currently, the llamafile server only serves one model at a time. However, there's no reason you can't start multiple llamafile servers on different ports. To support using different models for `smart_llm` and `fast_llm`, you could implement config vars like `LLAMAFILE_SMART_LLM_URL` and `LLAMAFILE_FAST_LLM_URL` that point to different llamafile servers (one serving a 'big model' and one serving a 'fast model'). 
* Authorization: the `serve.sh` script does not set up any authorization for the llamafile server; this can be turned on by adding arg `--api-key <some-key>` to the server startup command. However I haven't attempted to test whether the integration with autogpt works when this feature is turned on.
* Test with other models

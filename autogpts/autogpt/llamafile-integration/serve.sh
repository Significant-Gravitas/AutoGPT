#!/bin/bash
#
# Use llamafile to server a (quantized) mistral-7b-instruct-v0.2 model
#
# Usage:
#   cd <repo-root>/autogpts/autogpt
#   ./llamafile-integration/serve.sh
#

LLAMAFILE="./llamafile-integration/mistral-7b-instruct-v0.2.Q5_K_M.llamafile"

"${LLAMAFILE}" \
--server \
--nobrowser \
--ctx-size 0 \
--n-predict 1024

# note: ctx-size=0 means the prompt context size will be set directly from the
# underlying model configuration. This may cause slow response times or consume
# a lot of memory.
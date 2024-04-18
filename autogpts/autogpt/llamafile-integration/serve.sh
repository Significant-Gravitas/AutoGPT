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
--ctx-size 2048 \
--n-predict 512

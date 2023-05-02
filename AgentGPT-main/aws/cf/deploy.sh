#!/bin/bash
cd "$(dirname "$0")"

aws cloudformation create-stack --stack-name agent \
  --template-body file:///$PWD/agent.cf.json
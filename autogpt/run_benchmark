#!/bin/sh

# Kill processes using port 8080 if any.
if lsof -t -i :8080; then
    kill $(lsof -t -i :8080)
fi
# This is the cli entry point for the benchmarking tool.
# To run this in server mode pass in `serve` as the first argument.
poetry run agbenchmark "$@"

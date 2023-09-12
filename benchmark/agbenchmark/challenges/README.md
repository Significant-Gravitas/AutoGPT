# This is the official challenge library for https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks

The goal of this repo is to provide easy challenge creation for test driven development with the Auto-GPT-Benchmarks package. This is essentially a library to craft challenges using a dsl (jsons in this case).

This is the up to date dependency graph: https://sapphire-denys-23.tiiny.site/

### How to use

Make sure you have the package installed with `pip install agbenchmark`.

If you would just like to use the default challenges, don't worry about this repo. Just install the package and you will have access to the default challenges.

To add new challenges as you develop, add this repo as a submodule to your `project/agbenchmark` folder. Any new challenges you add within the submodule will get registered automatically.

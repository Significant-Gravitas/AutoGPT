# Next Gen AutoGPT 

This is a research project into creating the next generation of autogpt, which is an autogpt agent server.

It will come with the AutoGPT Agent as the default agent


## Project Outline

```
.
├── READEME.md
├── nextgenautogpt
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py              # The CLI tool for running the system 
│   ├── executor            # The Component Executor Process
│   │   └── __init__.py
│   ├── manager             # The Agent Manager it manages a pool of executors and schedules components to run
│   │   └── __init__.py
│   └── server              # The main application. It includes the api server and additional modules
│       └── __init__.py
└── pyproject.toml
```




# AutoGPT Core

This subpackage contains the ongoing work for the 
[AutoGPT Re-arch](https://github.com/Significant-Gravitas/AutoGPT/issues/4770). It is 
a work in progress and is not yet feature complete.  In particular, it does not yet
have many of the AutoGPT commands implemented and is pending ongoing work to 
[re-incorporate vector-based memory and knowledge retrieval](https://github.com/Significant-Gravitas/AutoGPT/issues/3536).

## [Overview](ARCHITECTURE_NOTES.md)

The AutoGPT Re-arch is a re-implementation of the AutoGPT agent that is designed to be more modular,
more extensible, and more maintainable than the original AutoGPT agent.  It is also designed to be
more accessible to new developers and to be easier to contribute to. The re-arch is a work in progress
and is not yet feature complete.  It is also not yet ready for production use.

## Running the Re-arch Code

1. Open the `autogpt/core` folder in a terminal

2. Set up a dedicated virtual environment:  
   `python -m venv .venv`

3. Install everything needed to run the project:  
   `poetry install`


## CLI Application

There are two client applications for AutoGPT included.

:star2: **This is the reference application I'm working with for now** :star2: 

The first app is a straight CLI application.  I have not done anything yet to port all the friendly display stuff from the ~~`logger.typewriter_log`~~`user_friendly_output` logic.  

- [Entry Point](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_app/cli.py)
- [Client Application](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_app/main.py)

You'll then need a settings file.  Run

```
poetry run cli make-settings
```

This will write a file called `default_agent_settings.yaml` with all the user-modifiable 
configuration keys to `~/auto-gpt/default_agent_settings.yml` and make the `auto-gpt` directory 
in your user directory if it doesn't exist). Your user directory is located in different places 
depending on your operating system:

- On Linux, it's `/home/USERNAME`
- On Windows, it's `C:\Users\USERNAME`
- On Mac, it's `/Users/USERNAME`

At a bare minimum, you'll need to set `openai.credentials.api_key` to your OpenAI API Key to run 
the model.

You can then run AutoGPT with 

```
poetry run cli run
```

to launch the interaction loop.

### CLI Web App

:warning: I am not actively developing this application.  I am primarily working with the traditional CLI app
described above.  It is a very good place to get involved if you have web application design experience and are 
looking to get involved in the re-arch.

The second app is still a CLI, but it sets up a local webserver that the client application talks to
rather than invoking calls to the Agent library code directly.  This application is essentially a sketch 
at this point as the folks who were driving it have had less time (and likely not enough clarity) to proceed.

- [Entry Point](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_web_app/cli.py)
- [Client Application](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_web_app/client/client.py)
- [Server API](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/core/runner/cli_web_app/server/api.py)

To run, you still need to generate a default configuration.  You can do 

```
poetry run cli-web make-settings
```

It invokes the same command as the bare CLI app, so follow the instructions above about setting your API key.

To run, do 

```
poetry run cli-web client
```

This will launch a webserver and then start the client cli application to communicate with it.

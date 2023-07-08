# Run instructions

There are two client applications for Auto-GPT included. 

## CLI Application

:star2: **This is the reference application I'm working with for now** :star2: 

The first app is a straight CLI application.  I have not done anything yet to port all the friendly display stuff from the `logger.typewriter_log` logic.  

- [Entry Point](https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/core/runner/cli_app/cli.py)
- [Client Application](https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/core/runner/cli_app/main.py)

Auto-GPT must be installed in your python environment to run this application.  To do so, run

```
pip install -e REPOSITORY_ROOT
```

where `REPOSITORY_ROOT` is the root of the Auto-GPT repository on your machine.

You'll then need a settings file.  Run

```
 python REPOSITORY_ROOT/autogpt/core/runner/cli_app/cli.py make-settings
 ```

This will write a file called `default_agent_settings.yaml` with all the user-modifiable configuration keys to `~/auto-gpt/default_agent_settings.yml` and make the `auto-gpt` directory in your user directory if it doesn't exist).  At a bare minimum, you'll need to set `openai.credentials.api_key` to your OpenAI API Key to run the model.

You can then run Auto-GPT with 

```
python REPOSITORY_ROOT/autogpt/core/runner/cli_app/cli.py run
```

to launch the interaction loop.

## CLI Web App

The second app is still a CLI, but it sets up a local webserver that the client application talks to rather than invoking calls to the Agent library code directly.  This application is essentially a sketch at this point as the folks who were driving it have had less time (and likely not enough clarity) to proceed.

- [Entry Point](https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/core/runner/cli_web_app/cli.py)
- [Client Application](https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/core/runner/cli_web_app/client/client.py)
- [Server API](https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/core/runner/cli_web_app/server/api.py)

To run, you still need to generate a default configuration.  You can do 

```
python REPOSITORY_ROOT/autogpt/core/runner/cli_web_app/cli.py make-settings
```

It invokes the same command as the bare CLI app, so follow the instructions above about setting your API key.

To run, do 

```
python REPOSITORY_ROOT/autogpt/core/runner/cli_web_app/cli.py client
```

This will launch a webserver and then start the client cli application to communicate with it.

:warning: I am not actively developing this application.  It is a very good place to get involved if you have web application design experience and are looking to get involved in the re-arch.
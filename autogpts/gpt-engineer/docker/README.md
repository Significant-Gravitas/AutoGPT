# Getting Started using Docker

**Running using docker cli**:

Building the image:
- `git clone https://github.com/AntonOsika/gpt-engineer.git`
- `cd gpt-engineer`
- `docker build --rm -t gpt-engineer -f docker/Dockerfile .`

Running the container:
- `docker run -it --rm -e OPENAI_API_KEY="YOUR OPENAI KEY" -v ./your-project:/project gpt-engineer`

The `-v` flag mounts the `your-project` folder into the container. Make sure to have a `prompt` file in there.

**Running using docker-compose cli**:

Building the image:
- `git clone https://github.com/AntonOsika/gpt-engineer.git`
- `cd gpt-engineer`
- `docker-compose -f docker-compose.yml build`
- `docker-compose run --rm gpt-engineer`


Set the OPENAI_API_KEY in docker/docker-compose.yml using .env file or environment variable, and mount your project folder into the container using volumes. for example "./projects/example:/project" ./projects/example is the path to your project folder.

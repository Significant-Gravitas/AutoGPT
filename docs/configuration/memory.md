## Setting Your Cache Type

By default, Auto-GPT is going to use LocalCache instead of redis or Pinecone.

To switch to either, change the `MEMORY_BACKEND` env variable to the value that you want:

* `local` (default) uses a local JSON cache file
* `pinecone` uses the Pinecone.io account you configured in your ENV settings
* `redis` will use the redis cache that you configured
* `milvus` will use the milvus cache that you configured
* `weaviate` will use the weaviate cache that you configured

## Memory Backend Setup

Links to memory backends

- [Pinecone](https://www.pinecone.io/)
- [Milvus](https://milvus.io/) &ndash; [self-hosted](https://milvus.io/docs), or managed with [Zilliz Cloud](https://zilliz.com/)
- [Redis](https://redis.io)
- [Weaviate](https://weaviate.io)

### Redis Setup
> _**CAUTION**_ \
This is not intended to be publicly accessible and lacks security measures. Therefore, avoid exposing Redis to the internet without a password or at all
1. Install docker (or Docker Desktop on Windows).
2. Launch Redis container.

``` shell    
    docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
```
> See https://hub.docker.com/r/redis/redis-stack-server for setting a password and additional configuration.

3. Set the following settings in `.env`.
    > Replace **PASSWORD** in angled brackets (<>)
    
``` shell
MEMORY_BACKEND=redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<PASSWORD>
```

    You can optionally set `WIPE_REDIS_ON_START=False` to persist memory stored in Redis.

You can specify the memory index for redis using the following:
``` shell
MEMORY_INDEX=<WHATEVER>
```

### 🌲 Pinecone API Key Setup

Pinecone lets you store vast amounts of vector-based memory, allowing the agent to load only relevant memories at any given time.

1. Go to [pinecone](https://app.pinecone.io/) and make an account if you don't already have one.
2. Choose the `Starter` plan to avoid being charged.
3. Find your API key and region under the default project in the left sidebar.

In the `.env` file set:
- `PINECONE_API_KEY`
- `PINECONE_ENV` (example: _"us-east4-gcp"_)
- `MEMORY_BACKEND=pinecone`

Alternatively, you can set them from the command line (advanced):

For Windows Users:

``` shell
setx PINECONE_API_KEY "<YOUR_PINECONE_API_KEY>"
setx PINECONE_ENV "<YOUR_PINECONE_REGION>" # e.g: "us-east4-gcp"
setx MEMORY_BACKEND "pinecone"
```

For macOS and Linux users:

``` shell
export PINECONE_API_KEY="<YOUR_PINECONE_API_KEY>"
export PINECONE_ENV="<YOUR_PINECONE_REGION>" # e.g: "us-east4-gcp"
export MEMORY_BACKEND="pinecone"
```

### Milvus Setup

[Milvus](https://milvus.io/) is an open-source, highly scalable vector database to store huge amounts of vector-based memory and provide fast relevant search. And it can be quickly deployed by docker locally or as a cloud service provided by [Zilliz Cloud](https://zilliz.com/).

1. Deploy your Milvus service, either locally using docker or with a managed Zilliz Cloud database.
    - [Install and deploy Milvus locally](https://milvus.io/docs/install_standalone-operator.md)

    - <details><summary>Set up a managed Zilliz Cloud database <i>(click to expand)</i></summary>

      1. Go to [Zilliz Cloud](https://zilliz.com/) and sign up if you don't already have account.
      2. In the *Databases* tab, create a new database.
          - Remember your username and password
          - Wait until the database status is changed to RUNNING.
      3. In the *Database detail* tab of the database you have created, the public cloud endpoint, such as:
      `https://xxx-xxxx.xxxx.xxxx.zillizcloud.com:443`.
    </details>

2. Run `pip3 install pymilvus` to install the required client library.
    Make sure your PyMilvus version and Milvus version are [compatible](https://github.com/milvus-io/pymilvus#compatibility) to avoid issues.
    See also the [PyMilvus installation instructions](https://github.com/milvus-io/pymilvus#installation).

3. Update `.env`
    - `MEMORY_BACKEND=milvus`
    - One of:
      - `MILVUS_ADDR=host:ip` (for local instance)
      - `MILVUS_ADDR=https://xxx-xxxx.xxxx.xxxx.zillizcloud.com:443` (for Zilliz Cloud)

    *The following settings are **optional**:*
    - Set `MILVUS_USERNAME='username-of-your-milvus-instance'`
    - Set `MILVUS_PASSWORD='password-of-your-milvus-instance'`
    - Set `MILVUS_SECURE=True` to use a secure connection. Only use if your Milvus instance has TLS enabled.
      Setting `MILVUS_ADDR` to a `https://` URL will override this setting.
    - Set `MILVUS_COLLECTION` if you want to change the collection name to use in Milvus. Defaults to `autogpt`.

### Weaviate Setup
[Weaviate](https://weaviate.io/) is an open-source vector database. It allows to store data objects and vector embeddings from ML-models and scales seamlessly to billion of data objects. [An instance of Weaviate can be created locally (using Docker), on Kubernetes or using Weaviate Cloud Services](https://weaviate.io/developers/weaviate/quickstart). 
Although still experimental, [Embedded Weaviate](https://weaviate.io/developers/weaviate/installation/embedded) is supported which allows the Auto-GPT process itself to start a Weaviate instance. To enable it, set `USE_WEAVIATE_EMBEDDED` to `True` and make sure you `pip install "weaviate-client>=3.15.4"`. 

#### Install the Weaviate client

Install the Weaviate client before usage.

``` shell
$ pip install weaviate-client
```

#### Setting up environment variables

In your `.env` file set the following:

``` shell
MEMORY_BACKEND=weaviate
WEAVIATE_HOST="127.0.0.1" # the IP or domain of the running Weaviate instance
WEAVIATE_PORT="8080" 
WEAVIATE_PROTOCOL="http"
WEAVIATE_USERNAME="your username"
WEAVIATE_PASSWORD="your password"
WEAVIATE_API_KEY="your weaviate API key if you have one"
WEAVIATE_EMBEDDED_PATH="/home/me/.local/share/weaviate" # this is optional and indicates where the data should be persisted when running an embedded instance
USE_WEAVIATE_EMBEDDED=False # set to True to run Embedded Weaviate
MEMORY_INDEX="Autogpt" # name of the index to create for the application
```
 
## View Memory Usage

View memory usage by using the `--debug` flag :)


## 🧠 Memory pre-seeding
Memory pre-seeding allows you to ingest files into memory and pre-seed it before running Auto-GPT.

``` shell
# python data_ingestion.py -h 
usage: data_ingestion.py [-h] (--file FILE | --dir DIR) [--init] [--overlap OVERLAP] [--max_length MAX_LENGTH]

Ingest a file or a directory with multiple files into memory. Make sure to set your .env before running this script.

options:
  -h, --help               show this help message and exit
  --file FILE              The file to ingest.
  --dir DIR                The directory containing the files to ingest.
  --init                   Init the memory and wipe its content (default: False)
  --overlap OVERLAP        The overlap size between chunks when ingesting files (default: 200)
  --max_length MAX_LENGTH  The max_length of each chunk when ingesting files (default: 4000)

# python data_ingestion.py --dir DataFolder --init --overlap 100 --max_length 2000
```

In the example above, the script initializes the memory, ingests all files within the `Auto-Gpt/autogpt/auto_gpt_workspace/DataFolder` directory into memory with an overlap between chunks of 100 and a maximum length of each chunk of 2000.

Note that you can also use the `--file` argument to ingest a single file into memory and that data_ingestion.py will only ingest files within the `/auto_gpt_workspace` directory.

The DIR path is relative to the auto_gpt_workspace directory, so `python data_ingestion.py --dir . --init` will ingest everything in `auto_gpt_workspace` directory.

You can adjust the `max_length` and `overlap` parameters to fine-tune the way the documents are presented to the AI when it "recall" that memory:
- Adjusting the overlap value allows the AI to access more contextual information from each chunk when recalling information, but will result in more chunks being created and therefore increase memory backend usage and OpenAI API requests.
- Reducing the `max_length` value will create more chunks, which can save prompt tokens by allowing for more message history in the context, but will also increase the number of chunks.
- Increasing the `max_length` value will provide the AI with more contextual information from each chunk, reducing the number of chunks created and saving on OpenAI API requests. However, this may also use more prompt tokens and decrease the overall context available to the AI.

Memory pre-seeding is a technique for improving AI accuracy by ingesting relevant data into its memory. Chunks of data are split and added to memory, allowing the AI to access them quickly and generate more accurate responses. It's useful for large datasets or when specific information needs to be accessed quickly. Examples include ingesting API or GitHub documentation before running Auto-GPT.

⚠️ If you use Redis as your memory, make sure to run Auto-GPT with the `WIPE_REDIS_ON_START=False` in your `.env` file.

⚠️For other memory backends, we currently forcefully wipe the memory when starting Auto-GPT. To ingest data with those memory backends, you can call the `data_ingestion.py` script anytime during an Auto-GPT run. 

Memories will be available to the AI immediately as they are ingested, even if ingested while Auto-GPT is running.

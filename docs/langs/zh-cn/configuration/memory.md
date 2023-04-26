## 设置缓存类型

默认情况下，Auto-GPT将使用LocalCache而不是redis或Pinecone。

要切换到其中一种，请将`MEMORY_BACKEND`环境变量更改为您想要的值：

* `local`（默认）使用本地JSON缓存文件
* `pinecone` 使用您在ENV设置中配置的Pinecone.io帐户
* `redis` 将使用您配置的redis缓存
* `milvus` 将使用您配置的milvus缓存
* `weaviate` 将使用您配置的weaviate缓存

## 内存后端设置

内存后端链接

- [Pinecone](https://www.pinecone.io/)
- [Milvus](https://milvus.io/) － [自托管](https://milvus.io/docs)，或使用[Zilliz Cloud](https://zilliz.com/)提供的管理服务
- [Redis](https://redis.io)
- [Weaviate](https://weaviate.io)

### Redis设置

> _**警告**_ \
此设置不适合公开访问，并缺少安全措施。因此，请避免在没有密码或完全暴露Redis的情况下将其暴露在Internet上。

1. 安装docker（或Windows上的Docker Desktop）。
2. 启动Redis容器。

``` shell    
    docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
```
> 有关设置密码和其他配置的信息，请参见https://hub.docker.com/r/redis/redis-stack-server。

3. 在`.env`中设置以下设置。
    > 使用尖括号（<>）替换**PASSWORD**
    
``` shell
MEMORY_BACKEND=redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=<PASSWORD>
```

    您可以选择将`WIPE_REDIS_ON_START=False`设置为持久化存储在Redis中的内存。

您可以使用以下方式指定redis的内存索引：

``` shell
MEMORY_INDEX=<WHATEVER>
```

### 🌲 Pinecone API密钥设置

Pinecone允许您存储大量基于向量的记忆，使代理程序可以在任何给定时间仅加载相关记忆。

1. 前往[pinecone](https://app.pinecone.io/)，如果您还没有帐户，请注册一个帐户。
2. 选择“Starter”计划以避免被收费。
3. 在左侧边栏的默认项目下找到您的API密钥和区域。

在`.env`文件中设置：

- `PINECONE_API_KEY`
- `PINECONE_ENV`（例如：“us-east4-gcp”）
- `MEMORY_BACKEND=pinecone`

或者，您可以从命令行设置它们（高级）：

对于Windows用户：

``` shell
setx PINECONE_API_KEY "<YOUR_PINECONE_API_KEY>"
setx PINECONE_ENV "<YOUR_PINECONE_REGION>" # 例如: "us-east4-gcp"
setx MEMORY_BACKEND "pinecone"
```

对于macOS和Linux用户：

``` shell
export PINECONE_API_KEY="<YOUR_PINECONE_API_KEY>"
export PINECONE_ENV="<YOUR_PINECONE_REGION>" # 例如: "us-east4-gcp"
export MEMORY_BACKEND="pinecone"
```

### Milvus设置

[Milvus](https://milvus.io/)是一个开源的、高度可扩展的向量数据库，可用于存储大量基于向量的记忆，并提供快速的相关检索。它可以通过本地Docker快速部署，也可以作为由[Zilliz Cloud](https://zilliz.com/)提供的云服务。

1. 部署您的Milvus服务，可以使用本地的docker或者由Zilliz Cloud提供的云服务。
    - [安装和部署本地Milvus](https://milvus.io/docs/install_standalone-operator.md)

    - <details><summary>设置托管的Zilliz Cloud数据库<i>（点击以展开）</i></summary>

      1. 前往[Zilliz Cloud](https://zilliz.com/)，如果您还没有帐户，请注册一个帐户。
      2. 在“数据库”选项卡中，创建一个新的数据库。
          - 记住您的用户名和密码
          - 等待直到数据库状态变为RUNNING。
      3. 在您创建的数据库的“数据库详细信息”选项卡中，找到公共云端点，例如：
      `https://xxx-xxxx.xxxx.xxxx.zillizcloud.com:443`。
    </details>

2. 运行`pip3 install pymilvus`以安装所需的客户端库。确保您的PyMilvus版本和Milvus版本是[兼容的](https://github.com/milvus-io/pymilvus#compatibility)，以避免出现问题。也可以参考[PyMilvus安装说明](https://github.com/milvus-io/pymilvus#installation)。

3. 更新`.env`
    - `MEMORY_BACKEND=milvus`
    - 其中之一：
      - `MILVUS_ADDR=host:ip`（针对本地实例）
      - `MILVUS_ADDR=https://xxx-xxxx.xxxx.xxxx.zillizcloud.com:443`（针对Zilliz Cloud）

    *以下设置是可选的：*
    - 如果您想要更改在Milvus中使用的集合名称，请设置`MILVUS_COLLECTION`。默认为`autogpt`。
    - 如果您想要使用安全连接，请设置`MILVUS_SECURE=True`。仅在Milvus实例启用了TLS时使用。将`MILVUS_ADDR`设置为`https://` URL将覆盖此设置。
    - 如果您想要设置Milvus实例的用户名和密码，请设置`MILVUS_USERNAME='username-of-your-milvus-instance'`和`MILVUS_PASSWORD='password-of-your-milvus-instance'`。
### Weaviate设置
Weaviate是一个开源的向量数据库，可以存储数据对象和来自机器学习模型的向量嵌入，并可以无缝地扩展到数十亿个数据对象。可以在本地（使用Docker）、在Kubernetes上或使用Weaviate云服务上创建Weaviate实例。虽然仍处于实验阶段，但支持嵌入式Weaviate，这允许Auto-GPT进程本身启动Weaviate实例。要启用它，请将`USE_WEAVIATE_EMBEDDED`设置为`True`，并确保您安装了`pip install "weaviate-client>=3.15.4"`。

#### 安装 Weaviate 客户端

在使用前，请先安装Weaviate客户端。

``` shell
$ pip install weaviate-client
```

#### 设置环境变量

在您的`.env`文件中设置以下内容：

``` shell
MEMORY_BACKEND=weaviate
WEAVIATE_HOST="127.0.0.1" # 运行Weaviate实例的IP或域名
WEAVIATE_PORT="8080" 
WEAVIATE_PROTOCOL="http"
WEAVIATE_USERNAME="您的用户名"
WEAVIATE_PASSWORD="您的密码"
WEAVIATE_API_KEY="您的Weaviate API密钥（如果有）"
WEAVIATE_EMBEDDED_PATH="/home/me/.local/share/weaviate" # 这是可选的，表示在运行嵌入式实例时应将数据持久化到何处
USE_WEAVIATE_EMBEDDED=False # 设置为True以运行嵌入式Weaviate
MEMORY_INDEX="Autogpt" # 应用程序创建的索引名称
```

## 查看内存使用情况

使用`--debug`标志来查看内存使用情况 :)

## 🧠 内存预载入
内存预先载入允许你在运行 Auto-GPT 之前将文件加载到内存中进行预先载入。

``` shell
# python data_ingestion.py -h 
usage: data_ingestion.py [-h] (--file FILE | --dir DIR) [--init] [--overlap OVERLAP] [--max_length MAX_LENGTH]

将单个文件或包含多个文件的目录加载到内存中。在运行此脚本之前，请确保设置了您的 .env。

选项:
  -h, --help               显示此帮助信息并退出
  --file FILE              要加载的文件。
  --dir DIR                包含要加载的文件的目录。
  --init                   初始化内存并清除其内容（默认: False）
  --overlap OVERLAP        加载文件时每个块之间的重叠大小（默认: 200）
  --max_length MAX_LENGTH  加载文件时每个块的最大长度（默认: 4000）

# python data_ingestion.py --dir DataFolder --init --overlap 100 --max_length 2000
```

在上面的示例中，该脚本初始化内存，将`Auto-Gpt/autogpt/auto_gpt_workspace/DataFolder`目录中的所有文件加载到内存中，并设置每个块之间的重叠为100，每个块的最大长度为2000。

请注意，您也可以使用 `--file` 参数将单个文件加载到内存中，并且 `data_ingestion.py` 仅会将 `/auto_gpt_workspace` 目录中的文件加载到内存中。

DIR 路径是相对于 auto_gpt_workspace 目录的，因此 `python data_ingestion.py --dir . --init` 将加载 auto_gpt_workspace 目录中的所有内容。

您可以调整 `max_length` 和 `overlap` 参数以微调向 AI 展示文档的方式，当 AI “回想起”信息时，它可以访问更多的上下文信息：
- 调整重叠值允许 AI 访问更多的上下文信息，但会导致创建更多块，从而增加了内存后端使用和 OpenAI API 请求的数量。
- 减少 `max_length` 值将创建更多块，可以通过允许在上下文中保存更多消息历史来节省提示令牌，但也会增加块的数量。
- 增加 `max_length` 值将为 AI 提供更多的上下文信息，减少创建的块数，并节省 OpenAI API 请求。但是，这可能会使用更多的提示令牌，并减少 AI 可用的整体上下文信息。

内存预先载入是一种提高 AI 准确性的技术，通过将相关数据加载到内存中，划分数据块并加入内存，使 AI 可以快速访问它们并生成更准确的响应。它适用于大型数据集或需要快速访问特定信息的情况。例如，在运行 Auto-GPT 之前，将 API 或 GitHub 文档加载到内存中。

⚠️ 如果你将 Redis 用作内存，确保在 `.env` 文件中以 `WIPE_REDIS_ON_START=False` 的方式运行 Auto-GPT。

⚠️ 对于其他内存后端，我们当前在启动 Auto-GPT 时强制清空内存。要在这些内存后端中加载数据，可以在 Auto-GPT 运行期间随时调用 `data_ingestion.py` 脚本。

一旦数据被加载到内存中，它们将立即对 AI 可用，即使是在运行 Auto-GPT 期间加载的。

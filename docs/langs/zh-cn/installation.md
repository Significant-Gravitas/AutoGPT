# 💾 安装指南

## ⚠️ 配置 OpenAI API Key

在 [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) 获取您的 OpenAI API Key。

为了在 Auto-GPT 中使用 OpenAI API Key，您**必须**已经设置了计费（即已经有了付费账户）。

您可以在 [https://platform.openai.com/account/billing/overview](https://platform.openai.com/account/billing/overview) 上设置付费账户。

重要提示：强烈建议您在 [使用情况页面](https://platform.openai.com/account/usage) 上跟踪您的使用情况。您还可以在 [使用限制页面](https://platform.openai.com/account/billing/limits) 上设置您的支出限制。

![为了让 OpenAI API Key 生效，您需要在 OpenAI API > Billing 中设置付费账户](./imgs/openai-api-key-billing-paid-account.png)

**在继续之前，请确保您已经完成了此步骤。否则，什么都不会起作用！**

## 一般设置

1. 确保您已经安装了 [**requirements**](https://github.com/Significant-Gravitas/Auto-GPT#-requirements) 中列出的其中一种环境。

   _要执行以下命令，请通过导航到计算机上的一个文件夹并在文件夹路径顶部输入 `CMD`（或 Bash 或 Powershell），然后按 Enter 打开一个 CMD、Bash 或 Powershell 窗口。确保已安装适用于您的操作系统的 [Git](https://git-scm.com/downloads)。_

2. 使用 Git 克隆仓库，或下载 [最新稳定版本](https://github.com/Significant-Gravitas/Auto-GPT/releases/latest)（在页面底部，点击 `Source code (zip)`）。

``` shell
    git clone -b stable https://github.com/Significant-Gravitas/Auto-GPT.git
```

3. 进入您下载仓库的目录。

``` shell
    cd Auto-GPT
```

4. 配置 Auto-GPT：
   1. 在主 `Auto-GPT` 文件夹中找到名为 `.env.template` 的文件。由于它的前缀是点号，因此在某些操作系统中，此文件可能默认为隐藏状态。要显示隐藏文件，请按照您的特定操作系统的说明进行操作（例如，在 Windows 中，单击文件浏览器中的 "查看" 选项卡并选中 "隐藏的项目" 复选框；在 macOS 中，按下 Cmd + Shift + .）。
   2. 通过删除 `template` 扩展名来创建 `.env` 的副本。最简单的方法是在命令提示符/终端窗口中执行 `cp .env.template .env`。
   3. 在文本编辑器中打开 `.env` 文件。
   4. 找到一行写着 `OPENAI_API_KEY=` 的代码。
   5. 在 `=` 后面输入您的唯一的 OpenAI API Key（不包括任何引号或空格）。
   6. 输入您想要使用的其他服务的 API Key 或 Token。要激活和调整设置，请删除 `#` 前缀。
   7. 保存并关闭 `.env` 文件。

   您现在已经配置了 Auto-GPT。

   注意事项：

   - 请参阅[配置 OpenAI API Key](#openai-api-keys-configuration)以获取您的 OpenAI API Key。
   - 在 [ElevenLabs](https://elevenlabs.io) 获取您的 xi-api-key。您可以在该网站的 "Profile" 标签上查看您的 xi-api-key。
   - 如果您想在 Azure 实例上使用 GPT，请将 `USE_AZURE` 设置为 `True`，然后按照以下步骤操作：
     - 将 azure.yaml.template 重命名为 azure.yaml，并在 azure_model_map 部分提供相关模型的 azure_api_base、azure_api_version 和所有部署 ID：
      - fast_llm_model_deployment_id - 您的 gpt-3.5-turbo 或 gpt-4 部署 ID
      - smart_llm_model_deployment_id - 您的 gpt-4 部署 ID
      - embedding_model_deployment_id - 您的 text-embedding-ada-002 v2 部署 ID

``` shell
# 请将以下所有值都用双引号括起来
# 将尖括号(<>)中的字符串替换为您自己的ID
azure_model_map:
    fast_llm_model_deployment_id: "<my-fast-llm-deployment-id>"
        ...
```
详细信息可在[https://pypi.org/project/openai/](https://pypi.org/project/openai/)的`Microsoft Azure Endpoints`部分以及[learn.microsoft.com](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings?tabs=command-line)的嵌入模型部分中找到。
如果您使用的是Windows操作系统，可能需要安装[msvc-170](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)。

4. 按照以下说明使用[Docker](#run-with-docker) (*推荐*)或[Docker-less](#run-docker-less)运行Auto-GPT。

### 使用Docker运行

最简单的方法是使用`docker-compose`运行：
``` shell
docker-compose build auto-gpt
docker-compose run --rm auto-gpt
```
默认情况下，这也将启动并连接Redis内存后端。
有关相关设置，请参见[Memory > Redis setup](./configuration/memory.md#redis-setup)。

您还可以使用“vanilla”docker命令进行构建和运行：
``` shell
docker build -t auto-gpt .
docker run -it --env-file=.env -v $PWD:/app auto-gpt
```

您可以传递额外的参数，例如，以`--gpt3only`和`--continuous`模式运行：
``` shell
docker-compose run --rm auto-gpt --gpt3only --continuous
```
``` shell
docker run -it --env-file=.env -v $PWD:/app --rm auto-gpt --gpt3only --continuous
```

或者，您可以直接从[Docker Hub](https://hub.docker.com/r/significantgravitas/auto-gpt)拉取最新版本并运行：
``` shell
docker run -it --env OPENAI_API_KEY='your-key-here' --rm significantgravitas/auto-gpt
```

或者将`ai_settings.yml`预设挂载：
``` shell
docker run -it --env OPENAI_API_KEY='your-key-here' -v $PWD/ai_settings.yaml:/app/ai_settings.yaml --rm significantgravitas/auto-gpt
```


### 不使用Docker运行

在终端中运行`./run.sh`（Linux/macOS）或`.\run.bat`（Windows）。这将安装任何必要的Python包并启动Auto-GPT。

### 使用Dev容器运行

1. 在VS Code中安装[Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)扩展程序。

2. 在命令面板中键入Dev Containers：Open Folder in Container。

3. 运行`./run.sh`。
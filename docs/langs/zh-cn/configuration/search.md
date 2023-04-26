## 🔍 配置 Google API Keys

注意：
此部分为可选项。如果您在运行 Google 搜索时遇到错误 429，请使用官方的 Google API。
要使用 `google_official_search` 命令，您需要在环境变量中设置您的 Google API 密钥。

创建项目：

1. 前往 [Google Cloud 控制台](https://console.cloud.google.com/)。
2. 如果您尚未拥有帐户，请创建一个并登录。
3. 点击页面顶部的“选择项目”下拉菜单，然后点击“新建项目”来创建新项目。
4. 给它一个名称并点击“创建”。
设置自定义搜索 API 并添加到您的 .env 文件：

5. 前往 [APIs & Services 仪表板](https://console.cloud.google.com/apis/dashboard)。
6. 点击“启用 APIs 和服务”。
7. 搜索“Custom Search API”并点击它。
8. 点击“启用”。
9. 前往 [凭据](https://console.cloud.google.com/apis/credentials) 页面。
10. 点击“创建凭据”。
11. 选择“API 密钥”。
12. 复制 API 密钥。
13. 在您的计算机上将其设置为名为 `GOOGLE_API_KEY` 的环境变量（请参阅如何设置环境变量）。
14. 在您的项目上[启用](https://console.developers.google.com/apis/api/customsearch.googleapis.com)自定义搜索 API。（可能需要等待几分钟以进行传播）
设置自定义搜索引擎并添加到您的 .env 文件：

15. 前往 [自定义搜索引擎](https://cse.google.com/cse/all) 页面。
16. 点击“添加”。
17. 按照提示设置您的搜索引擎。您可以选择搜索整个网络或特定站点。
18. 创建搜索引擎后，点击“控制面板”。
19. 点击“基础设置”。
20. 复制“搜索引擎 ID”。
21. 在您的计算机上将其设置为名为 `CUSTOM_SEARCH_ENGINE_ID` 的环境变量（请参阅如何设置环境变量）。

请记住，您的免费每日自定义搜索配额仅允许最多 100 次搜索。要增加此限制，您需要为项目分配一个计费帐户，以获得高达 10K 的每日搜索量。

### 设置环境变量

对于 Windows 用户：

```
setx GOOGLE_API_KEY "YOUR_GOOGLE_API_KEY"
setx CUSTOM_SEARCH_ENGINE_ID "YOUR_CUSTOM_SEARCH_ENGINE_ID"
```

对于 macOS 和 Linux 用户：

```
export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
export CUSTOM_SEARCH_ENGINE_ID="YOUR_CUSTOM_SEARCH_ENGINE_ID"
```

## 🖼 图像生成

默认情况下，Auto-GPT 使用 DALL-e 进行图像生成。要使用 Stable Diffusion，需要一个 [Hugging Face API Token](https://huggingface.co/settings/tokens)。

一旦您拥有令牌，请在您的 `.env` 中设置以下变量：

``` ini
IMAGE_PROVIDER=huggingface
HUGGINGFACE_API_TOKEN=YOUR_HUGGINGFACE_API_TOKEN
```

## Selenium
``` shell
sudo Xvfb :10 -ac -screen 0 1024x768x24 & DISPLAY=:10 <YOUR_CLIENT>
```

请注意，这段代码是用于在 Linux 环境下使用 Selenium 进行自动化测试时，设置虚拟显示器的命令。其中 `sudo Xvfb :10 -ac -screen 0 1024x768x24 &` 是启动虚拟显示器的命令，`DISPLAY=:10` 是设置环境变量，让 Selenium 程序使用虚拟显示器进行操作。`<YOUR_CLIENT>` 是您的 Selenium 程序。

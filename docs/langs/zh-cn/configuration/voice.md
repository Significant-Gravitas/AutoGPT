## 声音

输入以下命令以使用Auto-GPT的TTS（文本转语音）功能：

``` shell
python -m autogpt --speak
```

Eleven Labs提供语音技术，如语音设计、语音合成和预制语音，Auto-GPT可以使用这些技术来生成语音。

1. 前往 [Eleven Labs](https://beta.elevenlabs.io/) 并创建一个帐户（如果你还没有帐户）。
2. 选择并设置 `Starter` 计划。
3. 点击右上角的图标并找到“Profile”以查找你的API密钥。

在 `.env` 文件中设置以下值：
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_1_ID`（例如：`"premade/Adam"`）

以下是Eleven Labs提供的语音ID和名称列表。你可以使用名称或ID：

- Rachel：21m00Tcm4TlvDq8ikWAM
- Domi：AZnzlk1XvdvUeBnXmlld
- Bella：EXAVITQu4vr4xnSDxMaL
- Antoni：ErXwobaYiN019PkySvjV
- Elli：MF3mGyEYCl7XYWbV9V6O
- Josh：TxGEqnHWrfWFTfGW9XjX
- Arnold：VR6AewLTigWG4xSOukaG
- Adam：pNInz6obpgDQGcFmaJgB
- Sam：yoZ06aMxZJJ28mfd3POQ
- 
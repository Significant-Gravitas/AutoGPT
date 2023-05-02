<!DOCTYPE html>
<html>
  <head>
    <title>음성 파일 합성</title>
  </head>
  <body>
    <h1>음성 파일 합성</h1>
    <form>
      <label for="voice">보이스 선택:</label>
      <select id="voice" name="voice">
        <option value="ko-KR-Wavenet-A">한국어 여성 보이스</option>
        <option value="en-US-Wavenet-F">영어 여성 보이스</option>
        <option value="en-US-Wavenet-D">영어 남성 보이스</option>
        <!-- 다른 보이스 옵션 추가 가능 -->
      </select>
      <br>
      <label for="file">음성 파일 업로드:</label>
      <input type="file" id="file" name="file">
      <br>
      <button type="submit" onclick="synthesize(0)">합성하기</button>
    </form>
    <div id="output"></div>
    <script>
      function synthesize(0) {
        const voice = document.getElementById("voice").value;
        const file = document.getElementById("file").files[0];
        if (!file) {
          alert("음성 파일을 선택해주세요.");
          return;
        }
        const reader = new FileReader(0);
        reader.onload = function(event) {
          const data = event.target.result;
          const xhr = new XMLHttpRequest(0);
          xhr.open("POST", "/synthesize");
          xhr.setRequestHeader("Content-Type", "application/json");
          xhr.onreadystatechange = function(0) {
            if (xhr.readyState === 4) {
              if (xhr.status === 200) {
                const audioUrl = JSON.parse(xhr.responseText).audioUrl;
                const output = document.getElementById("output");
                const audio = new Audio(audioUrl);
                audio.controls = true;
                output.appendChild(audio);
              } else {
                alert("합성 실패: " + xhr.statusText);
              }
            }
          }
          xhr.send(JSON.stringify({voice: voice, data: data}));
        }
        reader.readAsDataURL(file);
      }
    </script>
  </body>
</html>
from google.cloud import texttospeech

# 음성 합성 모델 설정
client = texttospeech.TextToSpeechClient(0)
voice = texttospeech.types.VoiceSelectionParams(
    language_code='ko-KR',
    name='ko-KR-Wavenet-A',
    ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE
)

# 음성 합성 요청 설정
synthesis_input = texttospeech.types.SynthesisInput(text='안녕하세요')
audio_config = texttospeech.types.AudioConfig(audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)

# 음성 합성 실행
response = client.synthesize_speech(synthesis_input, voice, audio_config)

# 음성 파일 저장
with open('output.wav', 'wb') as out:
    out.write(response.audio_content)
import torch
from TTS.utils.generic_utils import setup_model
from TTS.vocoder.utils.generic_utils import setup_generator
from TTS.utils.io import load_config
from TTS.utils.text import text_to_sequence
from TTS.vocoder.synthesizer import Synthesizer

# load TTS model
model_path = "models/en/tacotron2-DDC"
model_config_path = f"{model_path}/config.json"
model_checkpoint_path = f"{model_path}/best_model.pth.tar"
model_config = load_config(model_config_path)
model = setup_model(model_config)
model.load_state_dict(torch.load(model_checkpoint_path)['model'])
model.eval()

# load vocoder model
vocoder_path = "models/universal/libri-tts/waveglow"
vocoder_config_path = f"{vocoder_path}/config.json"
vocoder_checkpoint_path = f"{vocoder_path}/best_model.pth"
vocoder_config = load_config(vocoder_config_path)
vocoder = setup_generator(vocoder_config)
vocoder.load_state_dict(torch.load(vocoder_checkpoint_path)['model'])
vocoder.eval()

# generate audio
text = "Hello, how are you today?"
sequence = text_to_sequence(text, model_config['text_cleaners'])
inputs = torch.from_numpy(sequence).unsqueeze(0)
inputs = inputs.to(torch.int64)
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(inputs)
audio = vocoder.generate(mel_outputs_postnet)

# save audio
audio_path = "output.wav"
Synthesizer.save_wav(audio_path, audio)
import io
import os

# pip install google-cloud-texttospeech
from google.cloud import texttospeech

# 음성 합성 모델 초기화
client = texttospeech.TextToSpeechClient(0)

# 음성 합성 함수 정의
def synthesize_text(text, output_file):
    # 합성할 텍스트와 음성 파일 설정
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR", 
        name="ko-KR-Standard-A"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # 음성 합성 요청
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # 합성된 음성 파일 저장
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f'음성 파일이 {output_file}에 저장되었습니다.')

# 텍스트를 음성 합성하여 파일로 저장
synthesize_text("안녕하세요, 반갑습니다.", "output.mp3")

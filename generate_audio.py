import os
from gtts import gTTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

text = "Tolong rapikan sandal atau sepatu anda."
output_path = os.path.join(AUDIO_DIR, "peringatan.mp3")

print(f"Generating audio warning: '{text}' ...")
tts = gTTS(text=text, lang="id", slow=False)
tts.save(output_path)
print(f"Audio saved successfully to: {output_path} (size: {os.path.getsize(output_path)} bytes)")

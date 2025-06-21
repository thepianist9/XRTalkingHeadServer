# --- 2. Imports ---
import base64, time, threading
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from funasr import AutoModel as FunASRModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from TTS.api import TTS
from IPython.display import Audio, Javascript, display
from google.colab import output

# --- Prepare event for synchronization ---
audio_ready_event = threading.Event()

# --- 3. Record 5 seconds of microphone audio in Colab ---
RECORD_JS = """
const sleep = time => new Promise(resolve => setTimeout(resolve, time));
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader();
  reader.onloadend = () => resolve(reader.result);
  reader.readAsDataURL(blob);
});
var record = async function() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  let data = [];
  recorder.ondataavailable = event => data.push(event.data);
  recorder.start();
  await sleep(5000);
  recorder.stop();
  await new Promise(resolve => recorder.onstop = resolve);
  const blob = new Blob(data, { type: 'audio/wav' });
  const b64 = await b2text(blob);
  google.colab.kernel.invokeFunction('notebook.receive_audio', [b64], {});
}
record();
"""

def receive_audio(b64_audio):
    header, b64 = b64_audio.split(",", 1)
    audio_bytes = base64.b64decode(b64)
    with open("mic_input.wav", "wb") as f:
        f.write(audio_bytes)
    print("✅ Audio saved as mic_input.wav")
    audio_ready_event.set()  # Signal main thread to continue

output.register_callback("notebook.receive_audio", receive_audio)

print("🎙 Speak now (recording for 5 seconds)...")
display(Javascript(RECORD_JS))

# Wait for audio to be saved or timeout after 15 seconds
if not audio_ready_event.wait(timeout=15):
    raise TimeoutError("Audio not received in time.")

start = time.perf_counter()

# --- 4. Run ASR ---
print("\n🧠 Transcribing speech...")

asr_result = asr_model.generate(
    input="mic_input.wav",
    cache={},
    language="en",
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
)
user_input = rich_transcription_postprocess(asr_result[0]["text"]).strip()
print("📝 Transcribed:", user_input)

# --- 5. Run LLM (Qwen) ---
print("\n💬 Generating response...")
device = "cuda" if torch.cuda.is_available() else "cpu"

messages = [
    {"role": "system", "content": "You are Tim, a 20-year-old who just celebrated his birthday. Reply naturally from Tim's perspective only."},
    {"role": "user", "content": user_input}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([prompt], return_tensors="pt").to(device)

output_ids = llm.generate(inputs.input_ids, max_new_tokens=50)
reply_ids = output_ids[0][inputs.input_ids.shape[1]:]
reply_text = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
print("🤖 Tim:", reply_text)

# --- 6. TTS ---
print("\n🔊 Synthesizing voice...")
wav = tts.tts(reply_text)
sf.write("tim_reply.wav", wav, 24000)

end = time.perf_counter()

# --- 7. Play the audio ---
print("🗣️ Tim says:")
print(f"Execution time: {end - start:.4f} seconds")
display(Audio("tim_reply.wav", autoplay=True))

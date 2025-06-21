# --- 2. Imports ---
import base64, time, threading
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from funasr import AutoModel as FunASRModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from IPython.display import Audio, Javascript, display
from google.colab import output
import asyncio
import edge_tts


def init():
    asr_model = FunASRModel(
    model="../ASR/SenseVoiceSmall",
    device="cuda:0",
    disable_update=True
    )
    llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat",
    torch_dtype="auto",
    device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")



if __name__ == "__main__":
    init()
import time
import sounddevice as sd
from faster_whisper import WhisperModel

t0 = time.time()
asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
print(f"Modello caricato in {time.time() - t0:.1f}s")

print("Parla per 5 secondi...")
audio = sd.rec(5 * 16000, samplerate=16000, channels=1, dtype="float32")
sd.wait()

t0 = time.time()
segs, _ = asr.transcribe(audio[:, 0], language="it", beam_size=1,
                         condition_on_previous_text=False, without_timestamps=True)
print("Testo:", " ".join(s.text.strip() for s in segs))
print(f"Trascrizione in {time.time() - t0:.2f}s")
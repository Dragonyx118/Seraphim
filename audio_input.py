"""Riceve l'audio del microfono dal Raspberry e lo trascrive con Whisper.

Protocollo (stesso formato che usa tts_kokoro.py verso il Raspberry):
    1 byte di tipo + 4 byte di lunghezza (big endian) + payload

    b"B"  inizio registrazione (il pulsante nell'app è stato premuto)
    b"A"  blocco audio: PCM int16 little endian, mono, 16000 Hz
    b"E"  fine registrazione (pulsante rilasciato / stream chiuso)

Il Raspberry apre una connessione per ogni registrazione: B, tanti A, E, poi chiude.
Se la connessione cade senza E, quello che è arrivato viene trascritto comunque.
"""

import socket
import struct
import threading

import numpy as np

SAMPLE_RATE = 16000

# Frasi inventate che Whisper tende a produrre sul rumore/silenzio
_ALLUCINAZIONI = ("amara.org", "sottotitoli creati dalla comunità")


def _recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class VoiceInput:
    def __init__(
        self,
        on_text,
        on_start=None,
        host="0.0.0.0",
        port=5006,
        model_size="small",
        device="cuda",
        compute_type="int8_float16",
        language="it",
        max_seconds=30,
        min_seconds=0.4,
    ):
        self.on_text = on_text
        self.on_start = on_start
        self.host = host
        self.port = port
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.max_bytes = int(max_seconds * SAMPLE_RATE * 2)
        self.min_samples = int(min_seconds * SAMPLE_RATE)

        self._model = None
        self._closed = threading.Event()
        self._srv = None

        # Il modello si carica in background, come Kokoro: non ritarda l'avvio
        threading.Thread(target=self._load_model, daemon=True).start()
        threading.Thread(target=self._serve, daemon=True).start()

    # ---------- API pubblica ----------

    def close(self):
        self._closed.set()
        if self._srv:
            try:
                self._srv.close()
            except OSError:
                pass

    # ---------- modello ----------

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[Whisper]: manca faster-whisper (pip install faster-whisper). Voce in ingresso disattivata.")
            return

        silenzio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        try:
            model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            # Il warm-up forza il caricamento delle librerie CUDA: se mancano, l'errore esce qui
            list(model.transcribe(silenzio, language=self.language)[0])
            print(f"[Whisper]: '{self.model_size}' pronto su {self.device}.")
        except Exception as exc:
            print(f"[Whisper]: {self.device} non disponibile ({exc}). Passo alla CPU.")
            try:
                model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                print(f"[Whisper]: '{self.model_size}' pronto su cpu.")
            except Exception as exc2:
                print(f"[Whisper]: impossibile caricare il modello ({exc2}).")
                return
        self._model = model

    def _transcribe(self, pcm):
        if self._model is None:
            print("[Whisper]: modello non ancora pronto, registrazione scartata.")
            return ""
        if len(pcm) % 2:
            pcm = pcm[:-1]
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if len(audio) < self.min_samples or np.abs(audio).max() < 0.01:
            return ""
        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if any(a in text.lower() for a in _ALLUCINAZIONI):
            return ""
        return text

    # ---------- rete ----------

    def _serve(self):
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(2)
            srv.settimeout(0.5)
            self._srv = srv
        except OSError as exc:
            print(f"[Voce]: impossibile ascoltare sulla porta {self.port} ({exc}).")
            return

        print(f"[Voce]: in ascolto sulla porta {self.port}.")
        while not self._closed.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle, args=(conn, addr), daemon=True
            ).start()

    def _handle(self, conn, addr):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(15)  # se il Pi sparisce senza chiudere, dopo 15 s si chiude la registrazione
        chunks = []
        n_bytes = 0
        try:
            while True:
                header = _recv_exact(conn, 5)
                if header is None:
                    break
                tipo = header[:1]
                (length,) = struct.unpack(">I", header[1:])
                if length > 1_000_000:  # nessun blocco audio ragionevole è così grande
                    break
                payload = _recv_exact(conn, length) if length else b""
                if payload is None:
                    break

                if tipo == b"B":
                    chunks, n_bytes = [], 0
                    if self.on_start:
                        try:
                            self.on_start()
                        except Exception as exc:
                            print(f"[Voce]: errore in on_start: {exc}")
                elif tipo == b"A":
                    if n_bytes < self.max_bytes:
                        chunks.append(payload)
                        n_bytes += len(payload)
                elif tipo == b"E":
                    break
        except OSError:
            pass  # timeout o connessione caduta: trascrivo quello che ho
        finally:
            conn.close()

        if not chunks:
            return
        try:
            text = self._transcribe(b"".join(chunks))
        except Exception as exc:
            print(f"[Whisper]: errore di trascrizione: {exc}")
            return
        if text:
            self.on_text(text)
        else:
            print("[Voce]: non ho capito niente (silenzio o rumore).")

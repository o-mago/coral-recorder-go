import sys
import os
import json
import wave
import time
import math
import numpy as np

# Audio Constants
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2 # 16-bit
CHUNK_SAMPLES = 7680 # ~0.48 seconds of audio
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH
RMS_SILENCE_THRESHOLD = 80.0
HANGOVER_CHUNKS = 4  # Keep active for ~1.92 seconds of silence

# Model Files (Standard and Core)
MODEL_DIR = "models"
MODEL_YAMNET_CORE = os.path.join(MODEL_DIR, "yamnet_core.vmfb") if os.path.exists(os.path.join(MODEL_DIR, "yamnet_core.vmfb")) else os.path.join(MODEL_DIR, "yamnet_core.tflite")
MODEL_COMMANDS_CORE = os.path.join(MODEL_DIR, "voice_commands_core.vmfb") if os.path.exists(os.path.join(MODEL_DIR, "voice_commands_core.vmfb")) else os.path.join(MODEL_DIR, "voice_commands_core.tflite")

MODEL_YAMNET = "models/yamnet.vmfb" if os.path.exists("models/yamnet.vmfb") else "models/yamnet.tflite"
MODEL_COMMANDS_CPU = "models/voice_commands.tflite"
MODEL_COMMANDS_TPU = "models/voice_commands.vmfb" if os.path.exists("models/voice_commands.vmfb") else "models/voice_commands_edgetpu.tflite"

# YAMNet Classes of Interest (based on Audioset Ontology)
CLASS_SPEECH = 0
CLASS_LAUGHTER = 25
CLASS_COUGH = 56
CLASS_CLAPPING = 68
CLASS_APPLAUSE = 69

# Coral Speech Commands v0.7 labels
COMMAND_LABELS = ['silence', 'unknown', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Optional Vosk speech recognition library for offline Portuguese voice commands
VOSK_AVAILABLE = False
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    pass

class IreeInterpreter:
    """Wrapper that mimics standard TFLite Interpreter API using iree.runtime for Synaptics Astra NPU."""
    def __init__(self, model_path):
        import iree.runtime as ireert
        driver = "hal"
        try:
            self.config = ireert.Config(driver)
        except Exception:
            driver = "local-task"
            self.config = ireert.Config(driver)
        
        sys.stderr.write(f"IREE/Torq Configured with driver: {driver}\n")
        self.ctx = ireert.SystemContext(config=self.config)
        with open(model_path, "rb") as f:
            flatbuffer_data = f.read()
            self.vm_module = ireert.VmModule.from_flatbuffer(self.config.vm_instance, flatbuffer_data)
        self.ctx.add_vm_module(self.vm_module)
        
        # Discover module name
        self.module_name = "module"
        for name in dir(self.ctx.modules):
            if not name.startswith("_") and name != "modules":
                self.module_name = name
                break
        
        self.invoke_func = getattr(self.ctx.modules, self.module_name)["main"]
        self.is_yamnet = "yamnet" in model_path.lower()
        self.input_data = None
        self.output_data = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        if self.is_yamnet:
            # Detect if it's the core model vs standard model
            # For the core model, shape is usually [1, 96, 64, 1] or similar
            if "core" in self.module_name.lower():
                return [{'index': 0, 'shape': (1, 96, 64, 1), 'dtype': np.float32, 'quantization': (1.0, 0.0)}]
            return [{'index': 0, 'shape': (15600,), 'dtype': np.float32, 'quantization': (1.0, 0.0)}]
        else:
            if "core" in self.module_name.lower():
                return [{'index': 0, 'shape': (1, 49, 10, 1), 'dtype': np.float32, 'quantization': (1.0, 0.0)}]
            return [{'index': 0, 'shape': (1, 16000), 'dtype': np.float32, 'quantization': (1.0, 0.0)}]

    def get_output_details(self):
        return [{'index': 0, 'dtype': np.float32, 'quantization': (1.0, 0.0)}]

    def set_tensor(self, index, data):
        self.input_data = data

    def invoke(self):
        raw_result = self.invoke_func(self.input_data)
        self.output_data = raw_result.to_host()

    def get_tensor(self, index):
        if hasattr(self.output_data, "ndim") and self.output_data.ndim == 1:
            return [self.output_data]
        elif isinstance(self.output_data, (list, tuple)) and not isinstance(self.output_data[0], (list, tuple, np.ndarray)):
            return [self.output_data]
        return self.output_data


def load_tflite_interpreter(model_path, num_threads=2):
    """Loads the TFLite interpreter (or IREE interpreter for .vmfb), trying to load with Edge TPU delegate first, fallback to CPU."""
    if model_path.endswith(".vmfb"):
        try:
            return IreeInterpreter(model_path)
        except Exception as e:
            sys.stderr.write(f"Failed to load IREE model {model_path}: {e}\n")
            return None

    tflite = None
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import ai_edge_litert.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                pass

    if tflite is not None:
        # Attempt to load using Edge TPU delegate if the model is compiled for Edge TPU
        if "edgetpu" in model_path:
            try:
                # Default path for libedgetpu on Linux/Coral Dev Boards
                if hasattr(tflite, "load_delegate"):
                    # Check first via ctypes if libedgetpu can be loaded to avoid ai-edge-litert bug
                    import ctypes
                    try:
                        ctypes.CDLL("libedgetpu.so.1")
                    except Exception as ctypes_err:
                        raise RuntimeError(f"libedgetpu.so.1 is not loadable: {ctypes_err}")
                    delegate = tflite.load_delegate("libedgetpu.so.1")
                    return tflite.Interpreter(model_path, experimental_delegates=[delegate], num_threads=num_threads)
                else:
                    raise AttributeError("load_delegate not found in module")
            except Exception as e:
                sys.stderr.write(f"Edge TPU not available for {model_path}, trying CPU: {e}\n")
                # Fall back to the CPU version of the model if available
                cpu_path = model_path.replace("_edgetpu", "")
                if os.path.exists(cpu_path):
                    return tflite.Interpreter(cpu_path, num_threads=num_threads)
                return None
        else:
            return tflite.Interpreter(model_path, num_threads=num_threads)
    else:
        sys.stderr.write("Warning: neither tflite_runtime, ai_edge_litert, nor tensorflow is installed.\n")
        return None


def calculate_rms(audio_data):
    """Calculates the RMS energy of an audio buffer (used as fallback for VAD/Commands)."""
    samples = np.frombuffer(audio_data, dtype=np.int16)
    if len(samples) == 0:
        return 0
    return math.sqrt(np.mean(samples.astype(np.float64)**2))


# --- NUMPY AUDIO FEATURE EXTRACTION ---

def get_mel_filterbank(num_mel_bins=64, num_spectrogram_bins=257, sample_rate=16000, lower_edge_hertz=125.0, upper_edge_hertz=7500.0):
    """Generates Mel filterbank matrix using numpy."""
    def hertz_to_mel(freq):
        return 1127.0 * np.log(1.0 + freq / 700.0)
    
    def mel_to_hertz(mel):
        return 700.0 * (np.exp(mel / 1127.0) - 1.0)
        
    mel_low = hertz_to_mel(lower_edge_hertz)
    mel_high = hertz_to_mel(upper_edge_hertz)
    mel_points = np.linspace(mel_low, mel_high, num_mel_bins + 2)
    hz_points = mel_to_hertz(mel_points)
    
    fft_freqs = np.linspace(0, sample_rate / 2.0, num_spectrogram_bins)
    
    fb = np.zeros((num_spectrogram_bins, num_mel_bins), dtype=np.float32)
    for i in range(num_mel_bins):
        left = hz_points[i]
        center = hz_points[i+1]
        right = hz_points[i+2]
        
        fb[:, i] = np.where(
            (fft_freqs >= left) & (fft_freqs <= right),
            np.where(
                fft_freqs <= center,
                (fft_freqs - left) / (center - left + 1e-6),
                (right - fft_freqs) / (right - center + 1e-6)
            ),
            0.0
        )
    return fb


def compute_log_mel_spectrogram(waveform, sample_rate=16000):
    """Computes the log-mel spectrogram of a waveform using numpy."""
    window_length = 400  # 25 ms
    hop_length = 160     # 10 ms
    fft_length = 512
    
    num_samples = len(waveform)
    if num_samples < window_length:
        return np.zeros((0, 64), dtype=np.float32)
        
    num_frames = 1 + (num_samples - window_length) // hop_length
    
    # Vectorized framing
    indices = np.arange(window_length)[None, :] + (np.arange(num_frames) * hop_length)[:, None]
    frames = waveform[indices]
    
    # Windowing (Periodic Hann window)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(window_length) / window_length)
    frames = frames * window
    
    # FFT Magnitude (STFT)
    rfft = np.fft.rfft(frames, n=fft_length, axis=-1)
    magnitudes = np.abs(rfft)  # shape: (num_frames, 257)
    
    # Mel Filterbank
    fb = get_mel_filterbank(num_mel_bins=64, num_spectrogram_bins=257, sample_rate=sample_rate, lower_edge_hertz=125.0, upper_edge_hertz=7500.0)
    mel_spectrogram = np.dot(magnitudes, fb)
    
    # Log scaling
    log_mel_spectrogram = np.log(mel_spectrogram + 0.001)
    return log_mel_spectrogram


def dct(x, num_coeffs=13):
    """Computes Discrete Cosine Transform (Type II) of x using numpy."""
    N = x.shape[-1]
    n = np.arange(N)
    k = np.arange(num_coeffs)[:, None]
    coeff_matrix = 2.0 * np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * N))
    return np.dot(x, coeff_matrix.T)


def compute_kws_features(waveform, num_frames=49, num_coefficients=10):
    """Computes KWS features (log-mel or MFCC) using numpy."""
    window_length = 480  # 30 ms
    num_samples = len(waveform)
    
    if num_frames > 1:
        hop_length = (num_samples - window_length) // (num_frames - 1)
        if hop_length <= 0:
            hop_length = 160
    else:
        hop_length = 160
        
    num_frames_actual = 1 + (num_samples - window_length) // hop_length
    
    # Vectorized framing
    indices = np.arange(window_length)[None, :] + (np.arange(num_frames_actual) * hop_length)[:, None]
    frames = waveform[indices]
    
    # Windowing (Periodic Hann window)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(window_length) / window_length)
    frames = frames * window
    
    # FFT Magnitude
    fft_length = 512
    rfft = np.fft.rfft(frames, n=fft_length, axis=-1)
    magnitudes = np.abs(rfft)
    
    # Mel Filterbank
    fb = get_mel_filterbank(num_mel_bins=40, num_spectrogram_bins=257, sample_rate=16000, lower_edge_hertz=20.0, upper_edge_hertz=4000.0)
    mel_spec = np.dot(magnitudes, fb)
    log_mel = np.log(mel_spec + 0.001)
    
    # Apply DCT to get MFCCs if num_coefficients < 40
    if num_coefficients < 40:
        features = dct(log_mel, num_coefficients)
    else:
        features = log_mel
        
    # Resize/interpolate to exactly num_frames if needed
    if len(features) != num_frames:
        x_old = np.linspace(0, 1, len(features))
        x_new = np.linspace(0, 1, num_frames)
        features_new = np.zeros((num_frames, num_coefficients), dtype=np.float32)
        for col in range(num_coefficients):
            features_new[:, col] = np.interp(x_new, x_old, features[:, col])
        features = features_new
        
    return features


def main():
    sys.stderr.write("Starting Coral Audio Processor in Python (Optimized for TPU/NPU)...\n")
    sys.stderr.write(f"CPU Optimization: Pre-filter VAD gate active (RMS Threshold: {RMS_SILENCE_THRESHOLD}, Hangover Chunks: {HANGOVER_CHUNKS})\n")
    sys.stderr.write("CPU Optimization: TFLite CPU thread count set to 2 (optimized for Dual-Core ARM Cortex-A55)\n")
    
    # 1. Load models
    interpreter_yamnet = None
    interpreter_commands = None
    vosk_recognizer = None
    mock_mode = False

    # Load Vosk Portuguese Model if available and configured
    if VOSK_AVAILABLE and os.path.exists("model"):
        try:
            sys.stderr.write("Loading Vosk Portuguese Speech Model...\n")
            vosk_model = Model("model")
            vosk_grammar = '["coral", "iniciar", "inicie", "começar", "comece", "gravar", "gravação", "parar", "terminar", "encerrar", "start", "stop", "[unk]"]'
            vosk_recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE, vosk_grammar)
            sys.stderr.write("Vosk Portuguese Speech Model loaded successfully!\n")
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to load Vosk model: {e}\n")

    # Path priorities for YAMNet
    if os.path.exists(MODEL_YAMNET_CORE):
        sys.stderr.write(f"Found NPU-compatible Core YAMNet at {MODEL_YAMNET_CORE}. Loading...\n")
        interpreter_yamnet = load_tflite_interpreter(MODEL_YAMNET_CORE)
        if interpreter_yamnet:
            interpreter_yamnet.allocate_tensors()
            sys.stderr.write("Core YAMNet model loaded successfully (Python feature extraction active)!\n")
    elif os.path.exists(MODEL_YAMNET):
        interpreter_yamnet = load_tflite_interpreter(MODEL_YAMNET)
        if interpreter_yamnet:
            interpreter_yamnet.allocate_tensors()
            sys.stderr.write("Standard YAMNet model loaded successfully (CPU fallback)!\n")
    else:
        sys.stderr.write(f"YAMNet model not found.\n")

    # Path priorities for Speech Commands
    if os.path.exists(MODEL_COMMANDS_CORE):
        sys.stderr.write(f"Found NPU-compatible Core Speech Commands at {MODEL_COMMANDS_CORE}. Loading...\n")
        interpreter_commands = load_tflite_interpreter(MODEL_COMMANDS_CORE)
        if interpreter_commands:
            interpreter_commands.allocate_tensors()
            sys.stderr.write("Core Speech Commands model loaded successfully (Python feature extraction active)!\n")
    else:
        cmd_model_path = MODEL_COMMANDS_TPU if os.path.exists(MODEL_COMMANDS_TPU) else MODEL_COMMANDS_CPU
        if os.path.exists(cmd_model_path):
            interpreter_commands = load_tflite_interpreter(cmd_model_path)
            if interpreter_commands:
                interpreter_commands.allocate_tensors()
                sys.stderr.write(f"Speech Commands model loaded from {cmd_model_path}!\n")
        else:
            sys.stderr.write("Speech Commands model not found.\n")

    # If neither Vosk nor TFLite voice commands are available, run in mock mode
    if not vosk_recognizer and not interpreter_commands:
        sys.stderr.write("RUNNING IN MOCK MODE (TFLite/Vosk models missing). Using energy-based signal detection.\n")
        mock_mode = True

    # 2. Configure audio buffers
    audio_window = np.zeros(16000, dtype=np.int16)
    
    recording = False
    wav_out = None
    
    # VAD Hangover parameters
    vad_hangover_chunks = 3
    vad_counter = 0
    yamnet_counter = 0

    # RMS Threshold for Mock VAD
    MOCK_RMS_THRESHOLD = 300.0 
    
    # Speech active counter for RMS pre-filter gate
    speech_active_counter = 0
    
    sys.stderr.write("Ready to receive audio stream over stdin...\n")

    try:
        while True:
            # Read exactly one chunk of audio from stdin
            raw_chunk = sys.stdin.buffer.read(CHUNK_BYTES)
            if not raw_chunk or len(raw_chunk) < CHUNK_BYTES:
                break # End of stream

            # Pre-filter energy check to skip silent chunks
            rms = calculate_rms(raw_chunk)
            if rms > RMS_SILENCE_THRESHOLD:
                speech_active_counter = HANGOVER_CHUNKS
            elif speech_active_counter > 0:
                speech_active_counter -= 1
                
            is_silent = (speech_active_counter == 0)

            # Convert read bytes to numpy int16 samples
            chunk_samples = np.frombuffer(raw_chunk, dtype=np.int16)
            
            # Slide the 1-second audio window
            audio_window = np.roll(audio_window, -len(chunk_samples))
            audio_window[-len(chunk_samples):] = chunk_samples

            # Normalize the window to float32 in [-1.0, 1.0] for models
            float_window = audio_window.astype(np.float32) / 32768.0

            # Initialize flags
            speech_detected = False
            start_command = False
            stop_command = False
            start_trigger_word = ""
            stop_trigger_word = ""
            text = ""

            # --- VOSK OFFLINE PORTUGUESE COMMAND DETECTION (guarded by VAD pre-filter) ---
            if vosk_recognizer:
                if not is_silent:
                    try:
                        # Feed the raw PCM chunk directly to Vosk
                        if vosk_recognizer.AcceptWaveform(raw_chunk):
                            res = json.loads(vosk_recognizer.Result())
                            text = res.get("text", "").lower()
                            if text:
                                sys.stderr.write(f"[Vosk Speech] Full Result: '{text}' (RMS: {rms:.1f})\n")
                        else:
                            res = json.loads(vosk_recognizer.PartialResult())
                            text = res.get("partial", "").lower()
                            if text:
                                sys.stderr.write(f"[Vosk Speech] Partial: '{text}' (RMS: {rms:.1f})\n")

                        text_clean = text.strip()
                        words = text_clean.split()
                        
                        if "coral" in words:
                            coral_idx = words.index("coral")
                            for distance in range(1, 3):
                                if coral_idx + distance < len(words):
                                    candidate = words[coral_idx + distance]
                                    
                                    start_keywords = ["iniciar", "inicie", "começar", "comecar", "comece", "gravar", "gravação", "gravacao", "start", "go"]
                                    if candidate in start_keywords:
                                        start_command = True
                                        start_trigger_word = f"coral + {candidate}"
                                        break
                                        
                                    stop_keywords = ["parar", "terminar", "encerrar", "stop"]
                                    if candidate in stop_keywords:
                                        stop_command = True
                                        stop_trigger_word = f"coral + {candidate}"
                                        break
                    except Exception as e:
                        sys.stderr.write(f"Vosk inference error: {e}\n")

            if mock_mode:
                if rms > MOCK_RMS_THRESHOLD:
                    speech_detected = True
                if not recording and rms > MOCK_RMS_THRESHOLD * 2:
                    start_command = True
            else:
                # --- SPEECH COMMANDS INFERENCE (Wake Word / Controls - Fallback to English TFLite) ---
                if interpreter_commands and not vosk_recognizer:
                    if not is_silent:
                        try:
                            input_details = interpreter_commands.get_input_details()
                            output_details = interpreter_commands.get_output_details()[0]

                            inp_details_0 = input_details[0]
                            is_core_commands = len(inp_details_0['shape']) > 1 and inp_details_0['shape'][0] != 16000 and inp_details_0['shape'][1] != 16000

                            if is_core_commands:
                                shape = inp_details_0['shape']
                                # Determine expected feature dimensions
                                if len(shape) == 4:
                                    num_frames, num_coeffs = shape[1], shape[2]
                                elif len(shape) == 3:
                                    num_frames, num_coeffs = shape[1], shape[2]
                                else:
                                    num_frames, num_coeffs = shape[0], shape[1]
                                    
                                features = compute_kws_features(float_window, num_frames=num_frames, num_coefficients=num_coeffs)
                                inp_data = features.reshape(shape)
                            else:
                                inp_data = float_window.reshape(inp_details_0['shape'])

                            if inp_details_0['dtype'] == np.int8:
                                scale, zero_point = inp_details_0['quantization']
                                inp_data = (inp_data / scale + zero_point).astype(np.int8)

                            interpreter_commands.set_tensor(inp_details_0['index'], inp_data)

                            if not is_core_commands and len(input_details) > 1:
                                inp_details_1 = input_details[1]
                                if inp_details_1['dtype'] == np.int32:
                                    sample_rate_val = np.array([SAMPLE_RATE], dtype=np.int32)
                                    interpreter_commands.set_tensor(inp_details_1['index'], sample_rate_val)

                            interpreter_commands.invoke()

                            cmd_out = interpreter_commands.get_tensor(output_details['index'])[0]
                            if output_details['dtype'] == np.int8:
                                scale, zero_point = output_details['quantization']
                                cmd_out = (cmd_out.astype(np.float32) - zero_point) * scale

                            max_idx = np.argmax(cmd_out)
                            prob = cmd_out[max_idx]
                            
                            if prob > 0.65:
                                predicted_word = COMMAND_LABELS[max_idx] if max_idx < len(COMMAND_LABELS) else 'unknown'
                                if predicted_word == 'go':
                                    start_command = True
                                elif predicted_word == 'stop':
                                    stop_command = True
                        except Exception as e:
                            if not hasattr(main, "_inference_error_logged"):
                                sys.stderr.write(f"Warning: Speech Commands inference failed: {e}\n")
                                main._inference_error_logged = True

                # --- YAMNET INFERENCE (VAD & Event Classification) ---
                if interpreter_yamnet and recording:
                    yamnet_counter += 1
                    if yamnet_counter % 2 == 0:
                        if not is_silent:
                            try:
                                input_details = interpreter_yamnet.get_input_details()[0]
                                output_details = interpreter_yamnet.get_output_details()[0]

                                is_core_yamnet = len(input_details['shape']) > 1 and input_details['shape'][0] != 15600

                                if is_core_yamnet:
                                    log_mel = compute_log_mel_spectrogram(float_window[:15600])
                                    yamnet_input = log_mel.reshape(input_details['shape'])
                                else:
                                    yamnet_input = float_window[:15600]

                                if input_details['dtype'] == np.int8:
                                    scale, zero_point = input_details['quantization']
                                    yamnet_input = (yamnet_input / scale + zero_point).astype(np.int8)
                                
                                interpreter_yamnet.set_tensor(input_details['index'], yamnet_input)
                                interpreter_yamnet.invoke()

                                yamnet_out = interpreter_yamnet.get_tensor(output_details['index'])[0]
                                if output_details['dtype'] == np.int8:
                                    scale, zero_point = output_details['quantization']
                                    yamnet_out = (yamnet_out.astype(np.float32) - zero_point) * scale
                                
                                if yamnet_out[CLASS_SPEECH] > 0.45:
                                    speech_detected = True

                                timestamp = time.strftime("%H:%M:%S")
                                if yamnet_out[CLASS_LAUGHTER] > 0.25:
                                    print(json.dumps({"type": "event", "value": "laughter", "timestamp": timestamp}), flush=True)
                                if yamnet_out[CLASS_CLAPPING] > 0.25 or yamnet_out[CLASS_APPLAUSE] > 0.25:
                                    print(json.dumps({"type": "event", "value": "clapping", "timestamp": timestamp}), flush=True)
                                if yamnet_out[CLASS_COUGH] > 0.25:
                                    print(json.dumps({"type": "event", "value": "cough", "timestamp": timestamp}), flush=True)
                            except Exception as e:
                                pass
                        else:
                            speech_detected = False
                    else:
                        speech_detected = True

            # --- RECORDER STATE MACHINE ---
            if start_command and not recording:
                recording = True
                wav_out = wave.open("meeting.wav", "wb")
                wav_out.setnchannels(CHANNELS)
                wav_out.setsampwidth(SAMPLE_WIDTH)
                wav_out.setframerate(SAMPLE_RATE)
                if vosk_recognizer:
                    vosk_recognizer.Reset()
                print(json.dumps({"type": "control", "value": "start"}), flush=True)
                if start_trigger_word:
                    sys.stderr.write(f"Recorder started via voice command: detected '{start_trigger_word}'\n")
                else:
                    sys.stderr.write("Recorder started via voice command!\n")

            if recording:
                if speech_detected:
                    vad_counter = vad_hangover_chunks
                elif not interpreter_yamnet and not mock_mode:
                    vad_counter = vad_hangover_chunks

                if vad_counter > 0:
                    wav_out.writeframes(raw_chunk)
                    vad_counter -= 1
                
                if stop_command:
                    recording = False
                    if wav_out:
                        wav_out.close()
                        wav_out = None
                    if vosk_recognizer:
                        vosk_recognizer.Reset()
                    print(json.dumps({"type": "control", "value": "done"}), flush=True)
                    if stop_trigger_word:
                        sys.stderr.write(f"Recorder stopped via voice command: detected '{stop_trigger_word}'\n")
                    else:
                        sys.stderr.write("Recorder stopped via voice command!\n")
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if wav_out:
            wav_out.close()
        sys.stderr.write("Coral Audio Processor shut down.\n")

if __name__ == "__main__":
    main()

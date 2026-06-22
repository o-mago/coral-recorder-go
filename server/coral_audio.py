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

# Model Files
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
            return [{'index': 0, 'shape': (15600,), 'dtype': np.float32, 'quantization': (1.0, 0.0)}]
        else:
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

def load_tflite_interpreter(model_path):
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
                    return tflite.Interpreter(model_path, experimental_delegates=[delegate])
                else:
                    raise AttributeError("load_delegate not found in module")
            except Exception as e:
                sys.stderr.write(f"Edge TPU not available for {model_path}, trying CPU: {e}\n")
                # Fall back to the CPU version of the model if available
                cpu_path = model_path.replace("_edgetpu", "")
                if os.path.exists(cpu_path):
                    return tflite.Interpreter(cpu_path)
                return None
        else:
            return tflite.Interpreter(model_path)
    else:
        sys.stderr.write("Warning: neither tflite_runtime, ai_edge_litert, nor tensorflow is installed.\n")
        return None

def calculate_rms(audio_data):
    """Calculates the RMS energy of an audio buffer (used as fallback for VAD/Commands)."""
    samples = np.frombuffer(audio_data, dtype=np.int16)
    if len(samples) == 0:
        return 0
    return math.sqrt(np.mean(samples.astype(np.float64)**2))

def main():
    sys.stderr.write("Starting Coral Audio Processor in Python...\n")
    
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
            vosk_recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
            sys.stderr.write("Vosk Portuguese Speech Model loaded successfully!\n")
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to load Vosk model: {e}\n")

    # Load YAMNet for event classification
    if os.path.exists(MODEL_YAMNET):
        interpreter_yamnet = load_tflite_interpreter(MODEL_YAMNET)
        if interpreter_yamnet:
            interpreter_yamnet.allocate_tensors()
            sys.stderr.write("YAMNet model loaded successfully!\n")
    else:
        sys.stderr.write(f"YAMNet model not found at {MODEL_YAMNET}.\n")

    # Load commands TPU model if available, else load CPU model
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
    # Maintain a sliding audio window (1 second = 16000 samples)
    audio_window = np.zeros(16000, dtype=np.int16)
    
    recording = False
    wav_out = None
    
    # VAD Hangover parameters (prevents choppy recording by keeping buffer active)
    vad_hangover_chunks = 3 # Keep recording active for ~1.5s after speech stops
    vad_counter = 0

    # RMS Threshold for Mock VAD
    MOCK_RMS_THRESHOLD = 300.0 
    
    sys.stderr.write("Ready to receive audio stream over stdin...\n")

    try:
        while True:
            # Read exactly one chunk of audio from stdin
            raw_chunk = sys.stdin.buffer.read(CHUNK_BYTES)
            if not raw_chunk or len(raw_chunk) < CHUNK_BYTES:
                break # End of stream

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

            # --- VOSK OFFLINE PORTUGUESE COMMAND DETECTION ---
            if vosk_recognizer:
                try:
                    # Feed the raw PCM chunk directly to Vosk
                    if vosk_recognizer.AcceptWaveform(raw_chunk):
                        res = json.loads(vosk_recognizer.Result())
                        text = res.get("text", "").lower()
                    else:
                        res = json.loads(vosk_recognizer.PartialResult())
                        text = res.get("partial", "").lower()

                    # Detect Portuguese keywords
                    if "iniciar gravação" in text or "iniciar gravacao" in text or ("iniciar" in text and "gravar" in text):
                        start_command = True
                    elif "parar gravação" in text or "parar gravacao" in text or "parar" in text or "finalizar" in text:
                        stop_command = True
                except Exception as e:
                    sys.stderr.write(f"Vosk inference error: {e}\n")

            if mock_mode:
                # Simplified logic based on RMS energy
                rms = calculate_rms(raw_chunk)
                
                # If energy exceeds the threshold, speech is detected
                if rms > MOCK_RMS_THRESHOLD:
                    speech_detected = True
                
                # Mock commands triggered by loud sound spikes
                if not recording and rms > MOCK_RMS_THRESHOLD * 2:
                    start_command = True
            else:
                # --- SPEECH COMMANDS INFERENCE (Wake Word / Controls - Fallback to English TFLite) ---
                if interpreter_commands and not vosk_recognizer:
                    try:
                        input_details = interpreter_commands.get_input_details()[0]
                        output_details = interpreter_commands.get_output_details()[0]

                        inp_data = float_window.reshape(input_details['shape'])
                        if input_details['dtype'] == np.int8:
                            scale, zero_point = input_details['quantization']
                            inp_data = (float_window / scale + zero_point).astype(np.int8).reshape(input_details['shape'])

                        interpreter_commands.set_tensor(input_details['index'], inp_data)
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
                        pass

                # --- YAMNET INFERENCE (VAD & Event Classification) ---
                if interpreter_yamnet:
                    try:
                        input_details = interpreter_yamnet.get_input_details()[0]
                        output_details = interpreter_yamnet.get_output_details()[0]

                        # YAMNet expects 15600 samples (0.975 seconds)
                        yamnet_input = float_window[:15600]
                        
                        interpreter_yamnet.set_tensor(input_details['index'], yamnet_input)
                        interpreter_yamnet.invoke()

                        yamnet_out = interpreter_yamnet.get_tensor(output_details['index'])[0]
                        
                        # 1. Voice Activity Detection (VAD)
                        if yamnet_out[CLASS_SPEECH] > 0.45:
                            speech_detected = True

                        # 2. Tag interesting acoustic events
                        timestamp = time.strftime("%H:%M:%S")
                        if yamnet_out[CLASS_LAUGHTER] > 0.25:
                            print(json.dumps({"type": "event", "value": "laughter", "timestamp": timestamp}), flush=True)
                        if yamnet_out[CLASS_CLAPPING] > 0.25 or yamnet_out[CLASS_APPLAUSE] > 0.25:
                            print(json.dumps({"type": "event", "value": "clapping", "timestamp": timestamp}), flush=True)
                        if yamnet_out[CLASS_COUGH] > 0.25:
                            print(json.dumps({"type": "event", "value": "cough", "timestamp": timestamp}), flush=True)
                    except Exception as e:
                        pass

            # --- RECORDER STATE MACHINE ---

            # Command to start recording
            if start_command and not recording:
                recording = True
                # Create the output WAV file
                wav_out = wave.open("meeting.wav", "wb")
                wav_out.setnchannels(CHANNELS)
                wav_out.setsampwidth(SAMPLE_WIDTH)
                wav_out.setframerate(SAMPLE_RATE)
                # Reset the Vosk recognizer to clean any pre-buffered speech
                if vosk_recognizer:
                    vosk_recognizer.Reset()
                # Notify Go
                print(json.dumps({"type": "control", "value": "start"}), flush=True)
                sys.stderr.write("Recorder started via voice command!\n")

            # If recording is active, handle VAD buffering and WAV write
            if recording:
                if speech_detected:
                    # If we don't have YAMNet, default VAD to True (continuous recording)
                    vad_counter = vad_hangover_chunks # Reset hangover counter
                elif not interpreter_yamnet and not mock_mode:
                    # In standard TFLite commands mode without YAMNet, record continuously
                    vad_counter = vad_hangover_chunks

                if vad_counter > 0:
                    # Write audio frames to the output WAV file
                    wav_out.writeframes(raw_chunk)
                    vad_counter -= 1
                
                # Command to stop recording
                if stop_command:
                    recording = False
                    if wav_out:
                        wav_out.close()
                        wav_out = None
                    if vosk_recognizer:
                        vosk_recognizer.Reset()
                    # Notify Go that recording is complete
                    print(json.dumps({"type": "control", "value": "done"}), flush=True)
                    sys.stderr.write("Recorder stopped via voice command!\n")
                    break

    except KeyboardInterrupt:
        pass
    finally:
        # Guarantee closure of WAV file if process is killed or aborted
        if wav_out:
            wav_out.close()
        sys.stderr.write("Coral Audio Processor shut down.\n")

if __name__ == "__main__":
    main()

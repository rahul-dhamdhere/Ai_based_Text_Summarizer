import whisper
import tempfile
import os
import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_audio(audio_data):
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name

        # Load the model and enable multi-GPU support
        model = whisper.load_model('small')
        if torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to('cuda')

        # Perform transcription
        result = model.module.transcribe(temp_audio_path, fp16=False) if torch.cuda.device_count() > 1 else model.transcribe(temp_audio_path, fp16=False)
        text = result['text']
        logging.info(f"Audio transcribed successfully: {text}")

        os.remove(temp_audio_path)
        return text
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        return f"Transcription error: {str(e)}"
import sounddevice as sd
import soundfile as sf
from io import BytesIO
from scipy import signal
import os
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=os.getenv('ELEVEN_LABS_API_KEY'))


def text_to_speech(text):
    try:
        # Generate audio
        audio = client.generate(
            text=text,
            voice="BXYepLXgEDL2bTMkcar4",  # Raphael voice
            model="eleven_multilingual_v2"
        )
        
        # Prepare audio data
        audio_data = b''.join(audio)
        data, sample_rate = sf.read(BytesIO(audio_data), dtype='float32')
        
        # Resample if necessary
        if sample_rate != device_sample_rate:
            number_of_samples = int(round(len(data) * float(device_sample_rate) / sample_rate))
            data = signal.resample(data, number_of_samples)
            sample_rate = device_sample_rate
        
        # Increase volume
        volume_increase = 10.0
        data = data * volume_increase
                
        # Play audio
        sd.play(data, samplerate=sample_rate, device=device_id)
        sd.wait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

text_to_speech("IM FUCKING GAY")

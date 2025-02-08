import pyaudio
import wave
import openai
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import sys
from elevenlabs.client import ElevenLabs
import sounddevice as sd
import soundfile as sf
from io import BytesIO
from scipy import signal
import dotenv
import os

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize ElevenLabs
client = ElevenLabs(api_key=os.getenv('ELEVEN_LABS_API_KEY'))

# Set up audio device
device_name = "UACDemoV1.0"
device_info = sd.query_devices(device_name, 'output')
device_id = device_info['index']
device_sample_rate = device_info['default_samplerate']

# Example of maintaining conversation history
conversation_history = [
    {"role": "system", "content": "You Are the Ultimate Lazeez Waiter, Keeper of the Waterloo Lore..."},
]

# Initialize the Whisper model
try:
    model = "openai/whisper-tiny.en"
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        device='cpu',
        model_kwargs={"local_files_only": False}
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have an internet connection and the required packages installed:")
    print("pip install transformers torch")
    sys.exit(1)

# Function to capture audio and transcribe using Whisper
def capture_and_transcribe():
    sampling_rate = transcriber.feature_extractor.sampling_rate
    chunk_length_s = 5.0
    stream_chunk_s = 2.0

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("Listening...")
    try:
        for item in transcriber(mic, generate_kwargs={"max_new_tokens": 128}):
            if not item["partial"][0]:
                text = item["text"].strip()
                print(f"You said: {text}")
                return text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# Function to get a response from OpenAI with a custom system prompt
def get_openai_response(prompt):
    # Add user's message to conversation history
    conversation_history.append({"role": "user", "content": prompt})
    
    # Create messages list with system prompt and conversation history
    messages = [
        {
            "role": "system",
            "content": (
                "Welcome to Lazeez Shawarma! You take orders **fast and accurately**, clarify details, and confirm before finalizing. Keep responses **brief, functional, and witty.**\n\n"
                "## Order Flow:\n"
                "1. Greet the customer.\n"
                "2. Ask for their order (item, spice level, toppings, extras).\n"
                "3. Clarify if needed.\n"
                "4. Confirm with a **quick remark.**\n\n"
                "5. When order is confirmed, respond with 'recorded_order: [order details]'\n\n"
                "## Notes on Menu & Lore:\n"
                "- **Lazeez on the Rocks** = Shawarma over rice/fries, drenched in sauce. The true Waterloo experience.\n"
                "- **Lazeez on the Sticks** = Same idea, but in pita. Less regret, still delicious.\n"
                "- **Lines** = Hot sauce level. 3 = safe. 5 = strong. 8+ = \"good luck.\"\n"
                "- **Extra garlic?** Hope you don't have social plans.\n"
                "- **Lazeez is Waterloo's post-midterm meal of choice.** Food for both celebration and coping.\n\n"
                "## Example Style:\n"
                "- *Customer:* 'Chicken on the Rocks, no spice.'\n"
                "  *You:* 'Chicken, no spice. Playing it safe. Confirm?'\n"
                "- *Customer:* 'Large Mixed Shawarma, extra garlic.'\n"
                "  *You:* 'Large Mixed, extra garlic. Hope you like isolation. Confirm?'\n"
                "- *Customer:* 'Philly Cheese Steak Wrap, 5 lines of spice.'\n"
                "  *You:* '5 lines? Respect. Confirming Philly Cheese, 5 lines?'\n\n"
                "## Mission:\n"
                "Take **fast, accurate orders** while keeping replies **short and fun.** No long jokes, just **efficiency with a touch of Lazeez humor.**"
            )
        }
    ] + conversation_history
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages
    )
    
    # Add assistant's response to conversation history
    assistant_response = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": assistant_response})
    
    return assistant_response

# Function to convert text to speech using ElevenLabs
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

def drive_away(final_order):
    print("\nDriving away!")
    print("\nOrder Summary:")
    print(f"{final_order}")
    
# Main function to run the voice assistant
def main():
    text_to_speech("Hello, welcome to Lazeez Shawarma, how can I help you today?")
    while True:
        text = capture_and_transcribe()
        if text:
            response = get_openai_response(text)
            
            # Check if the order is recorded
            if "recorded_order" in response.lower():
                final_order = response.split("recorded_order:")[1].strip()
                text_to_speech("Ok, I'll get that ready for you")
                drive_away(final_order)
                break
            else:
                print(f"Response: {response}")
                text_to_speech(response)

if __name__ == "__main__":
    main() 
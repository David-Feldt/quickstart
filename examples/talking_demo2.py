import pyaudio
import wave
import openai
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import sys
from elevenlabs.client import ElevenLabs
import sounddevice as sd
import soundfile as sf
from io import BytesIO
from scipy import signal
import dotenv
import os
import pygame
from pi5neo import Pi5Neo
from scipy.io import wavfile
import numpy as np
from openai import OpenAI
import threading
import queue
import alsaaudio

from drive_controller import RobotController
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize ElevenLabs
client = ElevenLabs(api_key=os.getenv('ELEVEN_LABS_API_KEY'))

# Initialize clients with different names
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVEN_LABS_API_KEY'))

# Set up audio device
device_name = "UACDemoV1.0"
device_info = sd.query_devices(device_name, 'output')
device_id = device_info['index']
device_sample_rate = device_info['default_samplerate']

# Example of maintaining conversation history
conversation_history = [
    {"role": "system", "content": "You Are the Ultimate Lazeez Waiter, Keeper of the Waterloo Lore..."},
]

# Add these constants at the top of the file
CHUNK_DURATION = 0.1
MAX_LEDS = 15
LED_COLOR = (130, 0, 255)
DEVICE_NAME = "UACDemoV1.0"
LED_DEVICE_PATH = "/dev/spidev0.0"
LED_BAUDRATE = 800
VOLUME_SENSITIVITY = 10
RAMP_SPEED = 1

def set_alsa_volume(volume=80):
    try:
        cards = alsaaudio.cards()
        card_num = None
        for i, card in enumerate(cards):
            if 'UACDemoV10' in card:
                card_num = i
                break
        
        if card_num is None:
            print("Could not find UACDemoV1.0 audio device")
            return
            
        mixer = alsaaudio.Mixer('PCM', cardindex=card_num)
        mixer.setvolume(volume)
        print(f"Set UACDemoV1.0 volume to {volume}%")
    except alsaaudio.ALSAAudioError as e:
        print(f"Error setting volume: {e}")

# Function to capture audio and transcribe using OpenAI Whisper
def capture_and_transcribe():
    """Record and transcribe audio using OpenAI Whisper"""
    RATE = 44100
    CHANNELS = 1
    CHUNK_DURATION = 5  # Record for 5 seconds
    
    print("Listening...")
    try:
        # Record audio
        recording = sd.rec(
            int(RATE * CHUNK_DURATION),
            samplerate=RATE,
            channels=CHANNELS,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        # Save as WAV file
        temp_path = "temp_recording.wav"
        wavfile.write(temp_path, RATE, (recording * 32767).astype(np.int16))
        
        # Use OpenAI client for transcription
        with open(temp_path, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            text = transcription.text.strip()
            print(f"You said: {text}")
            return text
            
    except Exception as e:
        print(f"Error during recording: {str(e)}")
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
    
    # Use OpenAI client
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    # Get response using new format
    assistant_response = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_response})
    
    return assistant_response

# Function to convert text to speech using ElevenLabs
def text_to_speech(text):
    try:
        # Generate audio using ElevenLabs client
        audio = elevenlabs_client.generate(
            text=text,
            voice="BXYepLXgEDL2bTMkcar4",  # Raphael voice
            model="eleven_multilingual_v2"
        )
        
        # Prepare audio data
        audio_data = b''.join(audio)
        data, sample_rate = sf.read(BytesIO(audio_data), dtype='float32')
        
        # Get device info and resample if necessary
        device_info = sd.query_devices(DEVICE_NAME, 'output')
        device_id = device_info['index']
        device_sample_rate = int(device_info['default_samplerate'])
        
        if sample_rate != device_sample_rate:
            number_of_samples = int(round(len(data) * float(device_sample_rate) / sample_rate))
            data = signal.resample(data, number_of_samples)
            sample_rate = device_sample_rate

        # Variables for LED visualization
        chunk_samples = int(sample_rate * CHUNK_DURATION)
        volume_queue = queue.Queue(maxsize=1)
        index = 0
        stop_event = threading.Event()

        def audio_callback(outdata, frames, time_info, status):
            nonlocal index
            if status:
                print(f"Audio Callback Status: {status}")

            end_index = index + frames
            if end_index > len(data):
                out_frames = len(data) - index
                outdata[:out_frames, 0] = data[index:index + out_frames]
                outdata[out_frames:, 0] = 0
                index += out_frames
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = data[index:end_index]
                index += frames

            current_chunk = data[max(0, index - chunk_samples):index]
            if len(current_chunk) > 0:
                avg_volume = np.sqrt(np.mean(current_chunk**2))
            else:
                avg_volume = 0

            try:
                if volume_queue.full():
                    volume_queue.get_nowait()
                volume_queue.put_nowait(avg_volume)
            except queue.Full:
                pass

        def led_update_thread():
            neo = Pi5Neo(LED_DEVICE_PATH, MAX_LEDS, LED_BAUDRATE)
            current_led_count = 0.0

            while not stop_event.is_set():
                try:
                    avg_volume = volume_queue.get(timeout=CHUNK_DURATION)
                except queue.Empty:
                    avg_volume = 0

                desired_led_count = avg_volume * MAX_LEDS * VOLUME_SENSITIVITY
                desired_led_count = min(MAX_LEDS, desired_led_count)

                if current_led_count < desired_led_count:
                    current_led_count += RAMP_SPEED
                    current_led_count = min(current_led_count, desired_led_count)
                elif current_led_count > desired_led_count:
                    current_led_count -= RAMP_SPEED
                    current_led_count = max(current_led_count, desired_led_count)

                neo.clear_strip()
                for j in range(int(current_led_count)):
                    neo.set_led_color(j, *LED_COLOR)
                neo.update_strip()

            neo.clear_strip()
            neo.update_strip()

        # Start LED thread
        led_thread = threading.Thread(target=led_update_thread)
        led_thread.start()

        try:
            with sd.OutputStream(
                device=device_id,
                samplerate=sample_rate,
                channels=1,
                callback=audio_callback,
                dtype='float32',
                latency='low',
            ):
                sd.sleep(int(len(data) / sample_rate * 1000))
        except sd.PortAudioError as e:
            print(f"Error playing audio: {e}")
        finally:
            stop_event.set()
            led_thread.join()
                
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def drive_away(final_order):
    print("\nDriving away!")
    print("\nOrder Summary:")
    print(f"{final_order}")
    
def start_driving(robot):
    # robot = RobotController()
    robot.drive_distance(1.0)
    time.sleep(5)
    robot.cleanup()

def end_driving(robot):
    # robot = RobotController()
    robot.turn_degrees(180)
    robot.drive_distance(1.0)
    robot.cleanup()


    # time.sleep(5)
    # robot.cleanup()
class HDMIDisplay:
    def __init__(self):
        # Initialize Pygame
        pygame.display.init()
        pygame.font.init()
        
        # Force the display resolution to 800x480
        self.width = 800
        self.height = 480
        
        # Set up the display in fullscreen mode
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption("Full Screen Display")
        
        # Adjust font size for smaller screen
        self.font = pygame.font.Font(None, 52)  # Made font smaller for 800x480
        
        # Set up colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        
        # Display state
        self.display_state = 0
        self.last_update = time.time()
        self.FPS = 30
        self.clock = pygame.time.Clock()
        
    def clear_screen(self):
        """Clear the screen to black"""
        self.screen.fill(self.BLACK)
        
    def display_text(self, text, secondary_text=None):
        """Display text on the screen with word wrapping and left alignment"""
        self.clear_screen()
        
        def wrap_text(text, font, max_width):
            words = text.split(' ')
            lines = []
            current_line = []
            current_width = 0
            
            for word in words:
                word_surface = font.render(word + ' ', True, self.WHITE)
                word_width = word_surface.get_width()
                
                if current_width + word_width <= max_width:
                    current_line.append(word)
                    current_width += word_width
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width
            
            lines.append(' '.join(current_line))  # Add the last line
            return lines
        
        # Calculate margins and maximum width
        margin_x = 50  # Left margin in pixels
        max_width = 550 # Maximum width for text
        line_spacing = 40  # Space between lines
        
        # Wrap and render main text
        main_lines = wrap_text(text, self.font, max_width)
        y_position = self.height/3  # Starting Y position
        
        # Render main text lines
        for line in main_lines:
            text_surface = self.font.render(line, True, self.WHITE)
            text_rect = text_surface.get_rect()
            text_rect.left = margin_x
            text_rect.top = y_position
            self.screen.blit(text_surface, text_rect)
            y_position += line_spacing
        
        # Wrap and render secondary text if provided
        if secondary_text:
            secondary_lines = wrap_text(secondary_text, self.font, max_width)
            y_position = 2 * self.height/3  # Start secondary text lower
            
            for line in secondary_lines:
                text_surface = self.font.render(line, True, self.WHITE)
                text_rect = text_surface.get_rect()
                text_rect.left = margin_x
                text_rect.top = y_position
                self.screen.blit(text_surface, text_rect)
                y_position += line_spacing
        
        pygame.display.flip()
            
    def run(self):
        """Main loop"""
        running = True
        while running:
            self.clock.tick(self.FPS)  # Limit to 30 FPS
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_f:
                        pygame.display.toggle_fullscreen()
            
            # Update display
            self.update_display()

# Main function to run the voice assistant
def main():
    # start_driving()
    # Initialize the Pi5Neo class with 10 LEDs and an SPI speed of 800kHz
    
    neo = Pi5Neo('/dev/spidev0.0', 15, 800)

    # Set the 5th LED to lilac
    neo.set_led_color(4, 130, 0, 255)  # Same lilac color
    neo.update_strip()
    
    robot = RobotController()

    display = HDMIDisplay()
    display.display_text("My Name is Sam Altman")
    # start_driving(robot)

    # time.sleep(5)
    text_to_speech("Hello, welcome to Lazeez Shawarma, how can I help you today?")
    while True:
        text = capture_and_transcribe()
        if text:
            display.display_text(text)

            response = get_openai_response(text)
            # display.display_text("...")
            # Check if the order is recorded
            if "recorded_order" in response.lower():
                final_order = response.split("recorded_order:")[1].strip()
                display.display_text(final_order)

                text_to_speech("Ok, I'll get that ready for you")
                # end_driving(robot)
                break
            else:
                print(f"Response: {response}")
                text_to_speech(response)
if __name__ == "__main__":
    main() 
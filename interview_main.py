import wave
import pyaudio
from pydub import AudioSegment
import asyncio
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig
import speech_recognition as sr
import ollama

# Set up constants for recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 30
WAVE_OUTPUT_FILENAME = "output.wav"

# Function to record audio interview
def record_interview():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save to a file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to process the recorded interview
async def process_interview():
    audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
    segment_length = len(audio) // 6 
    audio_segments = [audio[i * segment_length:(i + 1) * segment_length] for i in range(6)]

    emotions = []
    text_segments = []

    async def process_segment(segment, segment_index):
        segment_filename = f"output_segment_{segment_index}.wav"
        segment.export(segment_filename, format="wav") 

        client = HumeStreamClient("YOUR_HUME_API_KEY")
        config = ProsodyConfig()

        async with client.connect([config]) as socket:
            result = await socket.send_file(segment_filename)
            result = result['prosody']['predictions'][0]['emotions']
        
        top_3_emotions = sorted(result, key=lambda x: x['score'], reverse=True)[:3]
        emotions.append(top_3_emotions)

        recognizer = sr.Recognizer()
        with sr.AudioFile(segment_filename) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                text_segments.append(f"Text for segment {segment_index}: {text}")
            except sr.UnknownValueError:
                text_segments.append(f"Could not understand segment {segment_index}")
            except sr.RequestError:
                text_segments.append(f"API error in segment {segment_index}")

    tasks = [process_segment(segment, i) for i, segment in enumerate(audio_segments)]
    await asyncio.gather(*tasks)

    # Generate summary
    summary = generate_summary(emotions, text_segments, "What is a linked list?")
    return summary


def generate_summary(emotions, text_segments, question):
    prompt = f"""
    Summarize the user's interview based on the text and emotions. 
    Question: {question}

    {text_segments}

    Top emotions: {emotions}
    """
    output = ollama.generate(model="llama3.1", prompt=prompt)
    return output["response"]

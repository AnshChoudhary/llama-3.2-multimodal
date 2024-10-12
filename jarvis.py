import cv2
import speech_recognition as sr
import pyttsx3
import base64
from openai import OpenAI
from os import getenv
import threading
import time

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1", 
    api_key=getenv("FIREWORKS_API_KEY")
)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_image(image, prompt):
    image_base64 = image_to_base64(image)
    system_message = (
        "You are a helpful AI assistant named Jarvis. Your task is to analyze images and respond to prompts "
        "about them. Please provide informative and engaging responses based on what you see in the image. "
        "If you're unsure about something, it's okay to say so, but try to offer relevant observations or "
        "suggestions when possible. Avoid mentioning personal boundaries or discomfort unless the request "
        "is clearly inappropriate."
    )
    result = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
    )
    return result.choices[0].message.content

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen_for_wake_word(recognizer, microphone):
    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()
                if "jarvis" in text:
                    return True
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            print("Could not request results from speech recognition service")

def get_voice_input(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that. Could you please repeat?")
        return None
    except sr.RequestError:
        speak("Sorry, there was an error with the speech recognition service.")
        return None

def camera_thread(frame_buffer):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            frame_buffer[0] = frame
        time.sleep(0.1)  # Reduce CPU usage
    cap.release()

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    frame_buffer = [None]

    # Start camera thread
    threading.Thread(target=camera_thread, args=(frame_buffer,), daemon=True).start()

    speak("Hello, I'm Jarvis. Say my name when you need me.")

    while True:
        if listen_for_wake_word(recognizer, microphone):
            speak("How can I help you?")
            prompt = get_voice_input(recognizer, microphone)
            if prompt:
                speak("Processing your request. Please wait.")
                if frame_buffer[0] is not None:
                    response = process_image(frame_buffer[0], prompt)
                    speak("Here's what I found:")
                    speak(response)
                else:
                    speak("I'm sorry, but I couldn't capture an image. Please try again.")
            speak("Is there anything else I can help you with?")

if __name__ == "__main__":
    main()
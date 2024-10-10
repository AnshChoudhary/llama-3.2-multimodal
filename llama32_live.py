import cv2
import base64
from openai import OpenAI
from os import getenv
import numpy as np

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1", 
    api_key=getenv("FIREWORKS_API_KEY")
)

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_image(image, prompt):
    image_base64 = image_to_base64(image)
    result = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
        messages=[
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

def main():
    cap = cv2.VideoCapture(0)
    captured_image = None
    prompt = ""
    response = ""

    print("Controls:")
    print("Press 'c' to capture an image")
    print("Press 'p' to enter a prompt")
    print("Press 'r' to process the image")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Display captured image if available
        if captured_image is not None:
            cv2.imshow('Captured Image', captured_image)

        # Display prompt and response
        info = np.zeros((200, 3000, 3), np.uint8)
        cv2.putText(info, f"Prompt: {prompt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info, f"Response:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Split response into multiple lines
        y = 90
        for line in response.split('\n'):
            cv2.putText(info, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 20
            if y > 180:
                break

        cv2.imshow('Info', info)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            captured_image = frame.copy()
            print("Image captured!")
        elif key == ord('p'):
            prompt = input("Enter your prompt: ")
        elif key == ord('r'):
            if captured_image is not None and prompt:
                print("Processing image...")
                response = process_image(captured_image, prompt)
                print("Response received!")
            else:
                print("Please capture an image and enter a prompt first.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
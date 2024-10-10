import cv2
import base64
from openai import OpenAI
from os import getenv
import time

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

    print("Camera feed opened. Press 'q' in the camera window to close it.")
    print("\nInstructions:")
    print("1. Capture an image")
    print("2. Enter a prompt")
    print("3. Process the image")
    print("4. Quit the program")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            captured_image = frame.copy()
            cv2.imwrite('captured_image.png', captured_image)
            print("Image captured and saved as 'captured_image.png'")
        elif choice == '2':
            if captured_image is None:
                print("Please capture an image first.")
            else:
                prompt = input("Enter your prompt: ")
        elif choice == '3':
            if captured_image is None or 'prompt' not in locals():
                print("Please capture an image and enter a prompt first.")
            else:
                print("Processing image...")
                response = process_image(captured_image, prompt)
                print("\nModel Response:")
                print(response)
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
import cv2
import base64
from openai import OpenAI
from os import getenv
import time

def list_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def capture_image():
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("Error: No cameras detected.")
        print("Please ensure a camera is connected and not in use by another application.")
        return None
    
    print(f"Available cameras: {available_cameras}")
    camera_index = int(input("Enter the camera index to use: "))
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return None
    
    print("Initializing camera... Please wait.")
    time.sleep(2)  # Give the camera some time to initialize
    
    for _ in range(10):  # Try to capture a frame multiple times
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera Feed (Press q to capture)', frame)
            print("Camera feed opened. Press 'q' to capture an image.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to capture frame. Retrying...")
                    continue
                
                cv2.imshow('Camera Feed (Press q to capture)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.imwrite('screenshot.jpg', frame)
                    print("Image captured successfully.")
                    break
            break
        else:
            print("Failed to capture initial frame. Retrying...")
            time.sleep(1)
    else:
        print("Error: Failed to capture any frames after multiple attempts.")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    cap.release()
    cv2.destroyAllWindows()
    return 'screenshot.jpg'

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    client = OpenAI(
        base_url="https://api.fireworks.ai/inference/v1", 
        api_key=getenv("FIREWORKS_API_KEY")
    )

    # Capture image from camera
    try:
        image_path = capture_image()
        if image_path is None:
            print("Failed to capture image. Exiting.")
            return
    except Exception as e:
        print(f"Error capturing image: {e}")
        return

    # Get user input for the prompt
    user_prompt = input("Enter your prompt for the image: ")

    # Convert image to base64
    image_base64 = image_to_base64(image_path)

    result = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
    )

    # Print the model's response
    print(result.choices[0].message.content)

if __name__ == "__main__":
    main()
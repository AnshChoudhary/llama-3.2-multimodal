from openai import OpenAI
from pydantic import BaseModel, Field
import instructor
from os import getenv

class Model(BaseModel):
    model_name: str = Field(..., description="What is this model called?")
    model_use: str = Field(..., description="What is this deep learning architecture used for?")
    movie_working: str = Field(..., description="How does this model work?")

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.fireworks.ai/inference/v1", 
        api_key=getenv("FIREWORKS_API_KEY")
    ),
    mode=instructor.Mode.JSON
)

result = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
    response_model=Model,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the graph about and what are the "
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://cdn.boldbi.com/wp/blogs/unlocking-financial-insights/area-chart-example.webp"
                    }
                }
            ]
        }
    ],
)
print(result)


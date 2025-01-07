from openai import OpenAI
from pydantic import BaseModel, Field
import instructor
from os import getenv
from typing import List

class FinancialChartAnalysis(BaseModel):
    chart_title: str = Field(..., description="What is the title of the chart?")
    key_findings: List[str] = Field(..., description="Key findings from the chart analysis.")
    insights: str = Field(..., description="Overall insights and interpretation of the chart.")

client = instructor.from_openai(
    OpenAI(
        base_url="https://api.fireworks.ai/inference/v1", 
        api_key=getenv("FIREWORKS_API_KEY")
    ),
    mode=instructor.Mode.JSON)

result = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
    response_model=FinancialChartAnalysis,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is a financial chart. Please analyze it and provide the chart title, key findings, and overall insights."
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
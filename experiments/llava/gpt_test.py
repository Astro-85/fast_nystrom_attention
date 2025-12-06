from openai import OpenAI
from pathlib import Path
import base64, mimetypes

client = OpenAI()
data_url = "data:image/jpeg;base64," + base64.b64encode(Path("./experiments/llava/forest.jpg").read_bytes()).decode()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "What’s in this image?"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ],
)

print(response.choices[0].message.content)

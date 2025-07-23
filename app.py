import os
import httpx
import asyncio
import logging
from datetime import datetime
from openai import AzureOpenAI
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get secrets from .env
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Validate
if not AZURE_API_KEY or not AZURE_ENDPOINT:
    raise ValueError("Missing API key or endpoint. Check your .env file.")

# Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Config
DEPLOYMENT_NAME = "dall-e-3"
OUTPUT_DIR = os.path.join(os.curdir, "ad_images")
STANDARD_AD_SIZE = (1080, 1080)

# Azure client
client = AzureOpenAI(api_version="2024-02-01", api_key=AZURE_API_KEY, azure_endpoint=AZURE_ENDPOINT)

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def generate_prompt(product: str, audience: str) -> str:
    return (
        f"High-quality professional digital advertisement for {product}. "
        f"Target audience: {audience}. "
        f"Vibrant, modern, eye-catching, social media ready, centered composition."
    )

async def fetch_image(image_url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(image_url)
        response.raise_for_status()
        return response.content

async def generate_ad_image(prompt: str) -> bytes:
    try:
        result = client.images.generate(model=DEPLOYMENT_NAME, prompt=prompt, n=1)
        image_url = result.data[0].url
        return await fetch_image(image_url)
    except Exception as e:
        logging.error(f"Image generation failed: {e}")
        raise

def save_and_resize_image(image_bytes: bytes, filename_prefix="ad_image") -> str:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(STANDARD_AD_SIZE, Image.LANCZOS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path, format="PNG")
    return path

async def generate_ad(product: str, audience: str):
    ensure_output_dir()
    prompt = generate_prompt(product, audience)
    logging.info(f"Generating ad image for '{product}' targeting '{audience}'")

    image_data = await generate_ad_image(prompt)
    final_path = save_and_resize_image(image_data, filename_prefix=product.replace(" ", "_"))
    logging.info(f"Image saved at: {final_path}")
    return final_path

if __name__ == "__main__":
    asyncio.run(generate_ad(product="Greek Yogurt", audience="young urban fitness enthusiasts"))

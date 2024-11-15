from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import LanceDB
from langchain.indexes import SQLRecordManager, index

from langchain_aws import ChatBedrockConverse
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from typing import Literal, List
from PIL import Image, ImageFile
from pathlib import Path


import boto3
import json
import base64
import io


class ImagePrompt(BaseModel):
    type: Literal["interior", "exterior"]
    prompt: str


class ImagePrompts(BaseModel):
    list: List[ImagePrompt]


CSV_PATH = Path("./data/properties.csv")
COLLECTION_NAME = "property_listings"

VECSTORE_PATH = Path("vecstore")
VECSTORE_PATH.mkdir(parents=True, exist_ok=True)


def get_vector_store(path: str, csv_path: str) -> LanceDB:
    """Get the application vector store.

    This function creates a vector store from the CSV file at the given path.
    The vector store is created using the LanceDB backend and the OpenAIEmbeddings.
    The vector store is then indexed using the SQLRecordManager.

    Args:
        path (str): The path to the vector store.
        csv_path (str): The path to the CSV file.

    Returns:
        LanceDB: The vector store.
    """

    # Create the vector store
    vector_store = LanceDB(
        embedding=OpenAIEmbeddings(), uri=path, table_name=COLLECTION_NAME
    )

    # Load the data
    loader = CSVLoader(file_path=csv_path)

    # NOTE: Spliting data didn't help. It is better to keep full rows in the document.

    # Split the data
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    # docs = loader.load_and_split(text_splitter)
    docs = loader.load()

    # The record manager is used to keep an index of the documents already added
    record_manager = SQLRecordManager(
        f"lancedb/{COLLECTION_NAME}",
        db_url=f"sqlite:///{path}/record_manager_cache.sql",
    )
    record_manager.create_schema()

    # use index with a record manager to avoid data duplication
    index(docs, record_manager, vector_store, source_id_key="row")

    return vector_store


def gen_txt2img_prompts(description: str) -> List[ImagePrompt]:
    """Create image prompt instructions for a txt2img model based on the listing description"""

    llm = ChatBedrockConverse(
        model="mistral.mistral-large-2407-v1:0",
        temperature=0.0,
        max_tokens=2048,
    )

    parser = PydanticOutputParser(pydantic_object=ImagePrompts)
    format_instructions = parser.get_format_instructions()

    prompt = f"""
Given a property listing with details such as neighborhood, price, bedrooms, bathrooms, square footage, description, and neighborhood description, your task is to create detailed and specific prompts for an image generation model. These prompts should focus on visual aspects that can be depicted in an image, both for the interior and exterior of the property, as well as the surrounding neighborhood.

**Steps to follow:**

1. **Extract Relevant Information:** Identify key visual details from the property description, such as the style of the property, number of bedrooms and bathrooms, notable features (e.g., high ceilings, modern finishes), and any specific mentions of the neighborhood's characteristics.

2. **Create Interior Prompts:**
   - Generate prompts for each main room of the property (living room, bedroom, bathroom, kitchen, etc.).
   - Include details about the style, features, and overall ambiance of the rooms.
   - Ensure the prompts are high-quality, realistic, and include specifics like "4K" for image quality.

3. **Create Exterior Prompts:**
   - Generate prompts for the exterior of the property, describing its style and any notable features.
   - Include details about the neighborhood, such as the presence of restaurants, bars, shops, major highways, and public transportation.

4. **Write Clear and Detailed Prompts:** Each prompt should clearly describe the scene to be generated, avoiding details that cannot be visually represented (e.g., price, age of the property).

5. **Format Output:** Output the prompts in a structured format, with each prompt labeled as either "interior" or "exterior".

**Example Property Listing:**
```
Neighborhood: Midtown
Price: 500000.0
Bedrooms: 2
Bathrooms: 2
Sqft: 1500
Description: Beautiful contemporary style condo in the heart of Midtown. Only 5 years old, this property features an open floor plan, high ceilings, and modern finishes throughout.
Neighborhood Description: Midtown is a vibrant and bustling neighborhood with a mix of restaurants, bars, and shops. It is also conveniently located near public transportation and major highways.
```

**Example Output Prompts:**
1. "A high-quality, realistic 4K photo of a modern, contemporary style condo living room with an open floor plan, high ceilings, and modern finishes, featuring a large window with a view of the city."
2. "A high-quality, realistic 4K photo of a modern, contemporary style condo bedroom, with high ceilings and contemporary finishes, fit for a 2-bedroom, 2-bathroom condo in Midtown."
3. "A high-quality, realistic 4K photo of a modern, contemporary style condo bathroom, with high ceilings and modern finishes, fit for a 2-bedroom, 2-bathroom condo in Midtown."
4. "A high-quality, realistic 4K photo of a modern, contemporary style condo kitchen, with an open floor plan, high ceilings, and modern finishes, fit for a 2-bedroom, 2-bathroom condo in Midtown."
5. "A high-quality, realistic 4K photo of a vibrant and bustling neighborhood street scene with a mix of restaurants, bars, and shops, representing the Midtown neighborhood."
6. "A high-quality, realistic 4K photo of a bustling neighborhood with major highways and public transportation options in the distance, representing the Midtown neighborhood."
7. "A high-quality, realistic 4K photo of the exterior of a modern, contemporary style condo building, approximately 5 years old, located in the heart of a vibrant and bustling neighborhood like Midtown."

**Format Instructions:**
{format_instructions}

**Property Description:**
{description}
"""

    out = llm.invoke(prompt)
    out: ImagePrompts = parser.invoke(out)
    return out.list


def gen_image(prompt: str) -> ImageFile.ImageFile:
    client = boto3.client("bedrock-runtime")

    response = client.invoke_model(
        modelId="amazon.titan-image-generator-v2:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            dict(
                taskType="TEXT_IMAGE",
                textToImageParams=dict(
                    text=prompt,
                ),
                imageGenerationConfig=dict(
                    cfgScale=8,
                    numberOfImages=1,
                    quality="standard",
                    width=512,
                    height=512,
                    seed=1117389865,
                ),
            )
        ),
    )
    json_data = json.loads(response["body"].read())
    return Image.open(io.BytesIO(base64.b64decode(json_data["images"][0])))

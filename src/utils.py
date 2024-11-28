from langchain.indexes import SQLRecordManager, index
from langchain_chroma import Chroma
from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.tools import tool, create_retriever_tool, Tool
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.document_loaders import CSVLoader

from pydantic import BaseModel

from pathlib import Path
from typing import Literal

from typing import Literal, List, Sequence
from PIL import Image, ImageFile
from pathlib import Path

import regex as re
import pandas as pd
import boto3
import json
import base64
import io
import os


class ImagePrompt(BaseModel):
    type: Literal["interior", "exterior"]
    prompt: str


class ImagePrompts(BaseModel):
    list: List[ImagePrompt]


def get_vector_store(
    path: str,
    csv_path: str,
    collection_name: str,
    embeddings: Embeddings,
    source_id_key: str,
    metadata_columns: Sequence[str] = (),
) -> Chroma:
    """Get the application vector store.

    This function creates a vector store from the CSV file at the given path.
    The vector store is created using the LanceDB backend and the OpenAIEmbeddings.
    The vector store is then indexed using the SQLRecordManager.

    Args:
        path (str): The path to the vector store.
        csv_path (str): The path to the CSV file.
        collection_name (str): The name of the collection.
        embeddings (Embeddings): The embeddings model.
        source_id_key (str): The key for the source ID in the CSV metadata.
        metadata_columns (Sequence[str], optional): The columns to include as metadata. Defaults to ().

    Returns:
        Chroma: The vector store.
    """

    os.makedirs(path, exist_ok=True)

    # Create the vector store
    vector_store = Chroma(collection_name, embeddings, persist_directory=path)

    # Load the data
    loader = CSVLoader(file_path=csv_path, metadata_columns=metadata_columns)

    # Split the data
    # NOTE: Spliting data didn't help. It is better to keep full rows in the document.
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    # docs = loader.load_and_split(text_splitter)

    # The record manager is used to keep an index of the documents already added
    record_manager = SQLRecordManager(
        f"chroma/{collection_name}",
        db_url=f"sqlite:///{path}/record_manager_cache.sql",
    )
    record_manager.create_schema()

    # use index with a record manager to avoid data duplication
    index(loader.load(), record_manager, vector_store, source_id_key=source_id_key)

    return vector_store


def get_retriever_tool(
    property_listings: Chroma, with_row_number: bool = False
) -> Tool:
    listing_retriever = property_listings.as_retriever()
    if not with_row_number:
        return create_retriever_tool(
            listing_retriever,
            "retrieve_property_listings",
            "Queries and returns documents regarding property listings.",
        )

    @tool
    def retrieve_property_listings(query: str) -> str:
        """Queries and returns documents regarding property listings with row number.

        Args:
            query (str): The query string used to retrieve the most relevant documents

        Returns:
            str: The retrieved data including the row number
        """

        docs: List[Document] = listing_retriever.invoke(query)
        return "\n\n".join(
            f"Row: {doc.metadata["row"]}\n{doc.page_content}" for doc in docs
        )

    return retrieve_property_listings


def get_images_index(images_path: Path) -> pd.DataFrame:
    """Get a dataframe the listing image paths.

    Args:
        images_path (Path): Root path where the images are stored.
    """

    pattern = re.compile(r".*\/row_(?<row>\d+)\/pdx_(?<pdx>\d+)\.png")
    df = pd.DataFrame(columns=["id", "row", "pdx", "path"])

    for path in images_path.glob("**/*.png"):
        match = pattern.match(str(path))
        if match:
            row, pdx = match.groups()
            id = f"{int(row)}_{pdx}"
            df.loc[len(df)] = [id, int(row), int(pdx), str(path)]
    df.sort_values(by=["row", "pdx"], inplace=True)
    return df


def add_images_to_vector_store(vecstore: Chroma, df: pd.DataFrame) -> None:
    """Add the images to the vector store. Slower indexing and initialization. Bigger vector store.

    Args:
        vecstore (Chroma): Vector store with the CLIP embeddings function.
        df (pd.DataFrame): DataFrame with the listing image paths.
    """

    # Don't duplicate data
    dst_ids = vecstore.get(include=[])["ids"]
    df = df[~df["id"].isin(dst_ids)]

    if df.empty:
        return

    # Add images to vector store
    vecstore.add_images(
        df["path"].tolist(),
        df[["row", "pdx"]].to_dict(orient="records"),
        ids=df["id"].tolist(),
    )


def add_image_descriptions_to_vector_store(
    vecstore: Chroma, imgprompts_csv: Path
) -> None:
    """Add the image descriptions to the vector store. Faster indexing and initialization. Smaller vector store.

    Args:
        vecstore (Chroma): Vector store with the CLIP embeddings function.
        imgprompts_csv (Path): Path to the CSV with the image prompts, or image descriptions.
    """

    loader = CSVLoader(
        imgprompts_csv,
        metadata_columns=["id", "row", "pdx"],
        content_columns=["prompt"],
    )

    path = vecstore._persist_directory.strip("/")
    # The record manager is used to keep an index of the documents already added
    record_manager = SQLRecordManager(
        f"chroma/listing_images",
        db_url=f"sqlite:///{path}/record_manager_cache.sql",
    )
    record_manager.create_schema()

    # use index with a record manager to avoid data duplication
    index(
        loader.load(),
        record_manager,
        vecstore,
        cleanup="incremental",
        source_id_key="id",
    )


def add_listing_descriptions_to_vector_store(
    vecstore: Chroma, properties_csv: Path
) -> None:
    """Add listing descriptions to the vector store with the CLIP embeddings function.

    Args:
        vecstore (Chroma): Vector store with the CLIP embeddigngs function.
        properties_csv (Path): Path to the csv with the listing descriptions.
    """

    loader = CSVLoader(
        properties_csv, content_columns=["Description", "Neighborhood Description"]
    )

    path = vecstore._persist_directory.strip("/")
    # The record manager is used to keep an index of the documents already added
    record_manager = SQLRecordManager(
        f"chroma/listing_images",
        db_url=f"sqlite:///{path}/record_manager_cache.sql",
    )
    record_manager.create_schema()

    # use index with a record manager to avoid data duplication
    index(
        loader.load(),
        record_manager,
        vecstore,
        cleanup="incremental",
        source_id_key="row",
    )


def get_image_vector_store(
    vecstore_path: str,
    data_path: str,
    method: Literal["images", "descriptions"] = "descriptions",
) -> Chroma:
    """Get the vector store based on the method used to add data.

    Args:
        vecstore_path (str): Path to the vector store.
        data_path (str): Path to the data directory with the image paths or descriptions.
        method (Literal["images", "descriptions"]): Method used to add data to the vector store.

    Returns:
        Chroma: Vector store with the CLIP embeddings function.
    """

    embeddings = OpenCLIPEmbeddings(
        model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k"
    )
    vecstore = Chroma(
        persist_directory=vecstore_path,
        collection_name="images",
        embedding_function=embeddings,
    )

    if method == "images":
        df = get_images_index(data_path)
        add_images_to_vector_store(vecstore, df)
    elif method == "descriptions":
        add_image_descriptions_to_vector_store(vecstore, Path(data_path))
    else:
        raise ValueError("Invalid method. Use 'images' or 'descriptions'.")

    return vecstore


def get_retriever_by_image(images_vecstore: Chroma, listings_vecstore: Chroma) -> Tool:
    """Get the retriever by image.
    Args:
        images_vecstore (Chroma): Vector store with the CLIP embeddings function.
        listings_vecstore (Chroma): Vector store with the listings data.

    Returns:
        Tool: Retrieval tool for retrieving property listings by image.
    """

    @tool
    def retrieve_by_image(image_url: str) -> str:
        """Retrieve property listings by image"""

        images = images_vecstore.similarity_search_by_image(image_url)
        image_descriptions = [image.page_content for image in images]

        rows = set(int(image.metadata["row"]) for image in images)
        listing_descriptions: List[Document] = listings_vecstore.get(
            where={"row": {"$in": list(rows)}}
        )["documents"]

        result = []
        for row, listing in zip(rows, listing_descriptions):
            result.append(f"Row:{row}\n{listing}")

        return "\n\n".join(result)

    return retrieve_by_image


def gradio_add_listing_images(images_path: str, row: int) -> str:
    """Add the image of a property listing to the HTML page.

    Args:
        images_path (str): Path to the images folder.
        row (int): The row number of the listing to get the images.

    Returns:
        str: The formatted listing information as a HTML string.
    """

    image_tags = []
    row_path = Path(f"{images_path}/row_{row:05d}")

    if not row_path.exists():
        return "> No images found for this listing."

    for img_path in row_path.glob("*.png"):
        with open(img_path, "rb") as f:
            img_bytes = base64.b64encode(f.read())
            base64_string = img_bytes.decode("ascii")

        image_tags.append(
            f"""<img src="data:image/png;base64,{base64_string}" style="width: 100%; height: auto; aspect-ratio: 1 / 1;" />"""
        )

    return f"""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
{"\n".join(image_tags)}
</div>
<hr/>
"""


def gen_txt2img_prompts(description: str) -> List[ImagePrompt]:
    """Create image prompt instructions for a txt2img model based on the listing description.

    Args:
        description (str): The listing description to use for the image prompts.

    Returns:
        List[ImagePrompt]: The image prompts to use for the txt2img model.
    """

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


def gen_image(prompt: str, seed: int = 1117389865) -> ImageFile.ImageFile:
    """Generate an image based on the given prompt.

    Args:
        prompt (str): Prompt to generate an image with.
        seed (int): Seed for the random number generator.

    Returns:
        ImageFile.ImageFile: Generated image.
    """

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
                    seed=seed,
                ),
            )
        ),
    )
    json_data = json.loads(response["body"].read())
    return Image.open(io.BytesIO(base64.b64decode(json_data["images"][0])))

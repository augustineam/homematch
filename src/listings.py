from langchain_openai import OpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import CSVLoader

from dotenv import load_dotenv
from pydantic import BaseModel, Field, NonNegativeInt, NonNegativeFloat

from tqdm import tqdm
from typing import List
from pathlib import Path

from .agent import CSV_PATH
from .utils import gen_image, gen_txt2img_prompts

import click
import pandas as pd


load_dotenv(".env")

MODEL_NAME = "gpt-3.5-turbo-instruct"
TEMPERATURE = 0.7
MAX_TOKENS = 1000


class Property(BaseModel):
    """Propety listing information"""

    neighborhood: str = Field(description="Neighborhood were the property is located")
    price: NonNegativeFloat = Field(description="Price of the property listing")
    bedrooms: NonNegativeInt = Field(description="Number of bedrooms in the property")
    bathrooms: NonNegativeInt = Field(description="Number of bathrooms in the property")
    sqft: NonNegativeInt = Field(description="Square foots of the property")
    description: str = Field(
        description="The 1-2 sentence description of the property, including any relevant details"
    )
    neighborhood_desc: str = Field(
        description="THe 1-2 sentence description of the neighborhood, including any relevant details"
    )


class PropertyList(BaseModel):
    """List of property listings"""

    properties: List[Property]


@click.command()
@click.option("--num_properties", default=100, help="Number of properties to generate")
@click.option(
    "--output_path",
    default="./data/properties.csv",
    help="Path to save the generated properties",
)
@click.option(
    "--append/--no-append",
    default=True,
    help="Append to the existing file or overwrite it",
    is_flag=True,
    show_default=True,
)
def create_listings(num_properties: int, output_path: str, append: bool):
    """Create fake property listings"""

    def parse_output(text):
        output = parser.parse(text)
        # Return the list of properties as a list of dictionaries
        return [prop.model_dump() for prop in output.properties]

    # Create the model, the parser and the prompt
    llm = OpenAI(model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    parser = PydanticOutputParser(pydantic_object=PropertyList)
    format_instructions = parser.get_format_instructions()
    prompt = f"""
    You are a real estate expert. Your task is to create fake property listings based on the format instructions provided.
    The property description must be a 1-2 sentence description of the property, including style, age, and any other relevant details.
    The neighborhood description must be a 1-2 sentence description of the neighborhood, including any relevant details.
    Follow the format instructions close to the letter.

    {format_instructions}

    If you are providing a list, please create the list as follows `{{ properties: [{{property1}}, {{property2}}, ...] }}`

    ---

    Please generate the listings.
    """

    # Create a chain that will generate the listings and parse the output
    chain = llm | parse_output

    csv_path = Path(output_path)

    # If folder does not exist, create it
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True)

    # If file exists and append is False, delete it
    if csv_path.exists() and not append:
        csv_path.unlink()

    csv_mode = "a" if append else "w"

    columns = [
        "Neighborhood",
        "Price",
        "Bedrooms",
        "Bathrooms",
        "Sqft",
        "Description",
        "Neighborhood Description",
    ]

    with tqdm(total=num_properties) as pbar:
        while num_properties > 0:
            try:
                output = chain.invoke(prompt)
                out_df = pd.DataFrame(output)
                out_df.columns = columns

                # Save the dataframe to a csv file
                out_df.to_csv(
                    output_path,
                    mode=csv_mode,
                    header=not csv_path.exists(),
                    index=False,
                )
                num_properties -= len(out_df)
                pbar.update(len(out_df))
            except Exception as e:
                print(e)
                continue


from langchain_aws import BedrockEmbeddings


@click.command()
@click.option(
    "--csv-data",
    default=str(CSV_PATH),
    help="Path to the CSV property listings file",
    show_default=True,
)
def listing_images(csv_data: str):
    """Generate images for property listings"""

    path = Path(csv_data)

    loader = CSVLoader(file_path=path)
    docs = loader.load()

    img_path = path.parent / "images"
    img_path.mkdir(exist_ok=True)

    embedds_csv = path.parent / "images" / "embeddings.csv"
    total_docs = len(docs)

    embeddings = BedrockEmbeddings(model="amazon.titan-embed-text-v2:0")

    from langchain_community.vectorstores import LanceDB

    # We'll use this embeddings dataframe to store prompt embeddigns
    # and use it to copy an image generated already if the prompt is similar
    # to another one already generated
    df_embeddings = None
    # for doc in docs[44:]:
    for doc in docs[:44]:
        row = doc.metadata["row"]

        # row_imgs = img_path / f"row_{row:05d}"
        # row_imgs.mkdir(exist_ok=True)

        try:
            # Generate prompts for the listing
            prompts = gen_txt2img_prompts(doc.page_content)
            # total_prompts = len(prompts)

            for i, p in enumerate(prompts):
                print(f"{row + 1}/{total_docs} - {i + 1}/{len(prompts)}: {p}")

                df = pd.DataFrame(
                    data=dict(
                        row=row,
                        pdx=i,
                        type=p.type,
                        prompt=p.prompt,
                        embedding=[embeddings.invoke(p.prompt)],
                    )
                )

                df_embeddings = pd.concat([df_embeddings, df]) if df_embeddings else df
                df_embeddings.to_csv(
                    embedds_csv,
                    mode="a",
                    header=not embedds_csv.exists(),
                    index=False,
                )

            # # Generate images for each prompt
            # for i, p in enumerate(prompts):
            #     print(f"{row + 1}/{total_docs} - {i + 1}/{total_prompts}: {p.prompt}")
            #     image = gen_image(p.prompt)
            #     image.save(row_imgs / f"{p.type}_{i}.png")
        except:
            print(f"Error: generating image for listing at row {row}")

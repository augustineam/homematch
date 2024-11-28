from langchain_openai import OpenAI
from langchain_core.output_parsers import PydanticOutputParser

from dotenv import load_dotenv
from pydantic import BaseModel, Field, NonNegativeInt, NonNegativeFloat

from tqdm import tqdm
from typing import List
from pathlib import Path

from .utils import gen_txt2img_prompts, gen_image

import click
import random
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
        description="The 1-2 sentence description of the neighborhood, including any relevant details"
    )


class PropertyList(BaseModel):
    """List of property listings"""

    properties: List[Property]


@click.group()
def listings():
    """Commands for working with property listings"""
    pass


@click.command()
@click.option(
    "--num-properties",
    default=100,
    help="Number of properties to generate",
)
@click.option(
    "--properties-csv",
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
def create(num_properties: int, properties_csv: str, append: bool):
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

    csv_path = Path(properties_csv)

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
                    properties_csv,
                    mode=csv_mode,
                    header=not csv_path.exists(),
                    index=False,
                )
                num_properties -= len(out_df)
                pbar.update(len(out_df))
            except Exception as e:
                print(e)
                continue


@click.command()
@click.option(
    "--properties-csv",
    default="data/properties.csv",
    help="Path to the CSV property listings file",
    show_default=True,
)
@click.option(
    "--imgprompts-csv",
    default="data/imgprompts.csv",
    help="Path to the listings' image generation prompts",
    show_default=True,
)
def image_prompts(properties_csv: str, imgprompts_csv: str):
    """Create prompts for generating listing images"""

    properties_path = Path(properties_csv)
    if not properties_path.exists():
        raise FileNotFoundError(f"File {properties_csv} does not exist")

    imgprompts_path = Path(imgprompts_csv)

    properties_df = pd.read_csv(properties_path)
    imgprompts_df = pd.read_csv(imgprompts_path) if imgprompts_path.exists() else None

    # To continue from the last row of the imgprompts csv file
    offset_rows = imgprompts_df.iloc[-1]["row"] + 1 if imgprompts_path.exists() else 0

    for i, row in tqdm(
        properties_df.iloc[offset_rows:].iterrows(),
        initial=offset_rows,
        total=len(properties_df),
    ):

        try:
            # Format listing description as it if were a
            # vector store document
            doc_lines = []
            for k, v in row.to_dict().items():
                doc_lines.append(": ".join([k, str(v)]))
            listing_description = "\n".join(doc_lines)

            # Create the text to image prompts based on the
            # listing description
            prompts = gen_txt2img_prompts(listing_description)

            # Append each prompt to the end of the dataframe
            prompts_df = None
            for pdx, p in enumerate(prompts):
                rowdf = pd.DataFrame(
                    dict(id=f"{i}_{pdx}", row=i, pdx=pdx, **p.model_dump()), index=[0]
                )
                prompts_df = (
                    rowdf
                    if prompts_df is None
                    else pd.concat([prompts_df, rowdf], ignore_index=True)
                )

            # Save the prompts_df
            prompts_df.to_csv(
                imgprompts_csv,
                mode="a+",
                header=not imgprompts_path.exists(),
                index=False,
            )

        except Exception as e:
            print(f"Error ({e}): for row index {i}")


@click.command()
@click.option(
    "--prompts-csv",
    default="data/imgprompts.csv",
    help="Path to the listings' image generation prompts",
    show_default=True,
)
@click.option(
    "--images-path",
    default="data/images",
    help="Path to save generated images",
    show_default=True,
)
def generate_images(prompts_csv: str, images_path: str):
    """Generate images from text prompts"""

    prompts_path = Path("data/imgprompts.csv")
    if not prompts_path.exists():
        raise FileNotFoundError(f"File {prompts_csv} not found.")

    output_path = Path(images_path)
    output_path.mkdir(exist_ok=True)

    df_prompts = pd.read_csv(prompts_csv)

    row = 0
    seed = random.randint(0, 1000000)

    for _, row_data in tqdm(
        df_prompts.iterrows(),
        total=len(df_prompts),
        desc="Generating Images",
        unit="img",
    ):
        if row_data["row"] != row:
            row = row_data["row"]
            seed = random.randint(0, 1000000)
        pdx = row_data["pdx"]

        png_path = output_path / f"row_{row:05d}/pdx_{pdx}.png"
        png_path.parent.mkdir(exist_ok=True)

        if png_path.exists():
            continue

        try:
            image = gen_image(row_data["prompt"], seed)
            image.save(png_path)
        except Exception as e:
            print(f"Error ({e}): for row {row} and prompt {pdx}")


listings.add_command(create)
listings.add_command(image_prompts)
listings.add_command(generate_images)

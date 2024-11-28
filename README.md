# Personalized Real Estate Agent (HomeMatch)

## Introduction

This project aims to create a personalized real estate agent using AI for "Future Homes Realty", a forward-thinking real estate company. The project involves building a system that can understand user preferences, search for suitable properties, and provide recommendations based on those preferences.

## Project Structure

The project is structured into several key files:

- `main.py`: This is the main entry point of the application. It defines the CLI commands for interacting with the HomeMatch agent and managing property listings using the `click` library.

- `src/start.py`: This file contains the code for starting the HomeMatch agent in different modes (terminal, Gradio, and Langserve). It provides the `start` command with subcommands for each mode.

- `src/listings.py`: This file handles the creation of property listings. It also provides functions for creating fake property listings, generating image prompts, and generating the listings images.

- `src/agent.py`: This file defines the `HomeMatchAgent` function, which creates the personalized real estate agent. It includes methods for running the agent from the command line using an async streaming interface.

- `src/utils.py`: This file contains utility functions used throughout the project.

## Environment Setup

### Create Micromamba Environment

1. **Create a micromamba environment using the `environment.yml` file**:

   ```sh
   micromamba create -f environment.yml
   ```

2. **Activate the `torch` environment**:

   ```sh
   micromamba activate torch
   ```

3. **Install requirements from `requirements.txt`**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Create `.env` file with the following content**:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_API_BASE=your_openai_api_base_here
   ```

## Property Listings

### Download Fake Property Listings and Images

I have already generated property listings, image generation prompts and images using the methods described in following section, and can be downloaded following this link: [homematch_data.tar.gz](https://aam-genai-homematch.s3.us-west-2.amazonaws.com/homematch_data.tar.gz).

Once downloaded, create the `data/` folder and extract.

```sh
$ wget https://aam-genai-homematch.s3.us-west-2.amazonaws.com/homematch_data.tar.gz
$ mkdir data && cd data
$ tar xzvf ../homematch_data.tar.gz
```

### Generating Fake Property Listings and Images

The main script has a `listings` tool designed to generate a CSV file containing fake property listings. Here's how to use it:

```sh
Usage: main.py listings create [OPTIONS]

  Create fake property listings

Options:
  --num-properties INTEGER  Number of properties to generate
  --properties-csv TEXT     Path to save the generated properties
  --append / --no-append    Append to the existing file or overwrite it
                            [default: append]
  --help                    Show this message and exit.
```

The script uses the OpenAI language model to generate property listings based on predefined formats and descriptions. The generated listings are then saved to a CSV file.

**Example Output**

The generated CSV file (`properties.csv`) will contain columns such as `neighborhood`, `price`, `bedrooms`, `bathrooms`, `sqft`, `description`, and `neighborhood_desc`. Here is a sample of the output:

```csv
Neighborhood,Price,Bedrooms,Bathrooms,Sqft,Description,Neighborhood Description
Midtown,500000.0,2,2,1500,"Beautiful contemporary style condo in the heart of Midtown. Only 5 years old, this property features an open floor plan, high ceilings, and modern finishes throughout.","Midtown is a vibrant and bustling neighborhood with a mix of restaurants, bars, and shops. It is also conveniently located near public transportation and major highways."
Buckhead,750000.0,3,3,2500,"Stunning single-family home in the upscale neighborhood of Buckhead. This property offers a traditional design with a contemporary twist, including a gourmet kitchen and spacious master suite.","Buckhead is known for its luxurious homes, upscale shopping, and fine dining options. It is also home to several prestigious private schools and is a popular choice for families."
...
```

### Generate Fake Listing Images

In this section, two subcommands are introduced. One to create image prompts based on the property listings and the other to generate the fake listings images. I have used **AWS Bedrock**, and to reproduce the results it is adviced to add the AWS credentials to the file `.env` and request access to the models listed below in **AWS Bedrock**.

```sh
# .env
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_REGION="us-west-2"
```

The models used are:

- `mistral.mistral-large-2407-v1:0`: For image prompts
- `amazon.titan-image-generator-v2:0`: For image generation

Based on the property listings generated earlier, we can run the `listings image-prompts` subcommand to task an LLM to create prompts that we can use to task an image generator to create the images associated to the listings. This is how to use it:

```sh
Usage: main.py listings image-prompts [OPTIONS]

  Create prompts for generating listing images

Options:
  --properties-csv TEXT  Path to the CSV property listings file  [default:
                         data/properties.csv]
  --imgprompts-csv TEXT  Path to the listings' image generation prompts
                         [default: data/imgprompts.csv]
  --help                 Show this message and exit.
```

Then we can use the `listings generate-images` subcommand to iterate through the image prompts generated earlier and generate the fake image listings. This is how to use the subcommand.

```sh
Usage: main.py listings image-prompts [OPTIONS]

  Create prompts for generating listing images

Options:
  --properties-csv TEXT  Path to the CSV property listings file  [default:
                         data/properties.csv]
  --imgprompts-csv TEXT  Path to the listings' image generation prompts
                         [default: data/imgprompts.csv]
  --help                 Show this message and exit.
```

## Start the Agent

There are three ways to start interacting with the HomeMatch agent:

1. **Terminal**: You can interact with the agent directly in the terminal using the `start terminal` subcommand.
2. **Gradio**: The `start gradio` subcommand launches a web-based interface using Gradio, which provides a user-friendly way to chat with the agent. This is the preferred option as it allows you to see the fake listing images generated and upload your own images for the agent to find similar listings.
3. **Langserve**: The `start langserve` subcommand starts a langserve server, enabling interaction with the agent through an API.

Here's how to use the `start` command:

```sh
Usage: main.py start [OPTIONS] COMMAND [ARGS]...

  Start HomeMatch AI agent

Options:
  --vecstore TEXT  Path to vector store.  [default: ./vecstore]
  --trace          Shows the graph execution trace.
  --help           Show this message and exit.

Commands:
  gradio     Chat with the HomeMatch agent from the gradio UI.
  langserve  Serve HomeMatch agent as an API endopoint with LangServe and...
  terminal   Chat with the HomeMatch agent from the terminal.
```

### Screenshots from the Gradio ChatInterface

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; justify-items: center; margin: 0 auto;">
  <img src="images/screenshot-1.png" alt="screenshot-1">
  <img src="images/screenshot-2.png" alt="screenshot-2">
  <img src="images/screenshot-3.png" alt="screenshot-3">
  <img src="images/screenshot-4.png" alt="screenshot-4">
  <img src="images/screenshot-5.png" alt="screenshot-5">
</div>

from src.agent import chat
from src.listings import create_listings, listing_images

import click


@click.group()
def cli():
    pass


cli.add_command(create_listings)
cli.add_command(chat)
cli.add_command(listing_images)

if __name__ == "__main__":
    cli()

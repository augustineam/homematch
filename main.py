from src.agent import chat
from src.listings import listings

import click


@click.group()
def cli():
    pass


cli.add_command(chat)
cli.add_command(listings)

if __name__ == "__main__":
    cli()

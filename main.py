from src.start import start
from src.listings import listings

import click


@click.group()
@click.option(
    "--properties-csv",
    default="data/properties.csv",
    help="Path to the CSV property listings file",
    show_default=True,
)
@click.pass_context
def cli(ctx: click.Context, properties_csv: str):
    ctx.ensure_object(dict)
    ctx.obj["properties_csv"] = properties_csv


cli.add_command(start)
cli.add_command(listings)

if __name__ == "__main__":
    cli()

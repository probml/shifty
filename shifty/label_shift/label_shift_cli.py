import click

@click.command()
@click.option("--config_name", type=str, required=True)

def cli(
    config_name: str):
    pass

if __name__ == "__main__":
    #initialize_cache("jit_cache")
    cli()

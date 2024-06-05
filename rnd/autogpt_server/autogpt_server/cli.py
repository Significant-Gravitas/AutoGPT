import click


@click.group()
def main():
    """AutoGPT Server CLI Tool"""
    pass


@main.command()
def event():
    """
    Send an event to the running server
    """
    print("Event sent")


if __name__ == "__main__":
    main()

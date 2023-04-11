import click

# isort: off
from duckduckgo_search import (
    __version__,
    ddg,
    ddg_answers,
    ddg_images,
    ddg_maps,
    ddg_news,
    ddg_suggestions,
    ddg_translate,
    ddg_videos,
)

# isort: on

COLORS = {
    0: "black",
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "magenta",
    6: "cyan",
    7: "bright_black",
    8: "bright_red",
    9: "bright_green",
    10: "bright_yellow",
    11: "bright_blue",
    12: "bright_magenta",
    13: "bright_cyan",
}


def print_data(data):
    if data:
        for i, e in enumerate(data, start=1):
            click.secho(f"{i}. {'-' * 78}", bg="black", fg="white")
            for j, (k, v) in enumerate(e.items(), start=1):
                if v:
                    width = (
                        300
                        if k in ("href", "url", "image", "thumbnail", "content")
                        else 78
                    )
                    k = "language" if k == "detected_language" else k
                    text = click.wrap_text(
                        f"{v}",
                        width=width,
                        initial_indent="",
                        subsequent_indent=" " * 12,
                        preserve_paragraphs=True,
                    )
                else:
                    text = v
                click.secho(f"{k:<12}{text}", bg="black", fg=COLORS[j], overline=True)
            input()


@click.group(chain=True)
def cli():
    pass


@cli.command()
def version():
    print(__version__)
    return __version__


@cli.command()
@click.option("-k", "--keywords", help="text search, keywords for query")
@click.option(
    "-r",
    "--region",
    default="wt-wt",
    help="wt-wt, us-en, uk-en, ru-ru, etc. - search region https://duckduckgo.com/params",
)
@click.option(
    "-s",
    "--safesearch",
    default="moderate",
    type=click.Choice(["on", "moderate", "off"]),
    help="Safe Search",
)
@click.option(
    "-t",
    "--time",
    default=None,
    type=click.Choice(["d", "w", "m", "y"]),
    help="search results for the last day, week, month, year",
)
@click.option(
    "-m",
    "--max_results",
    default=25,
    help="maximum number of results, max=200, default=25",
)
@click.option(
    "-o",
    "--output",
    default="print",
    help="csv, json (save the results to a csv or json file)",
)
@click.option(
    "-d",
    "--download",
    is_flag=True,
    default=False,
    help="download results to 'keywords' folder",
)
def text(output, download, *args, **kwargs):
    data = ddg(output=output, download=download, *args, **kwargs)
    if output == "print" and not download:
        print_data(data)


@cli.command()
@click.option("-k", "--keywords", help="answers search, keywords for query")
@click.option(
    "-rt", "--related", default=False, is_flag=True, help="Add related topics"
)
@click.option(
    "-o",
    "--output",
    default="print",
    help="csv, json (save the results to a csv or json file)",
)
def answers(output, *args, **kwargs):
    data = ddg_answers(output=output, *args, **kwargs)
    if output == "print":
        print_data(data)


@cli.command()
@click.option("-k", "--keywords", help="keywords for query")
@click.option(
    "-r",
    "--region",
    default="wt-wt",
    help="wt-wt, us-en, uk-en, ru-ru, etc. - search region https://duckduckgo.com/params",
)
@click.option(
    "-s",
    "--safesearch",
    default="moderate",
    type=click.Choice(["on", "moderate", "off"]),
    help="Safe Search",
)
@click.option(
    "-t",
    "--time",
    default=None,
    type=click.Choice(["Day", "Week", "Month", "Year"]),
    help="search results for the last day, week, month, year",
)
@click.option(
    "-size",
    "--size",
    default=None,
    type=click.Choice(["Small", "Medium", "Large", "Wallpaper"]),
    help="",
)
@click.option(
    "-c",
    "--color",
    default=None,
    type=click.Choice(
        [
            "color",
            "Monochrome",
            "Red",
            "Orange",
            "Yellow",
            "Green",
            "Blue",
            "Purple",
            "Pink",
            "Brown",
            "Black",
            "Gray",
            "Teal",
            "White",
        ]
    ),
)
@click.option(
    "-type",
    "--type_image",
    default=None,
    type=click.Choice(["photo", "clipart", "gif", "transparent", "line"]),
)
@click.option(
    "-l", "--layout", default=None, type=click.Choice(["Square", "Tall", "Wide"])
)
@click.option(
    "-lic",
    "--license_image",
    default=None,
    type=click.Choice(["any", "Public", "Share", "Modify", "ModifyCommercially"]),
    help="""any (All Creative Commons), Public (Public Domain), Share (Free to Share and Use),
        ShareCommercially (Free to Share and Use Commercially), Modify (Free to Modify, Share,
        and Use), ModifyCommercially (Free to Modify, Share, and Use Commercially)""",
)
@click.option(
    "-m",
    "--max_results",
    default=100,
    help="maximum number of results, max=1000, default=100",
)
@click.option(
    "-o",
    "--output",
    default="print",
    help="csv, json (save the results to a csv or json file)",
)
@click.option(
    "-d",
    "--download",
    is_flag=True,
    default=False,
    help="download and save images to 'keywords' folder",
)
def images(output, download, *args, **kwargs):
    data = ddg_images(output=output, download=download, *args, **kwargs)
    if output == "print" and not download:
        print_data(data)


@cli.command()
@click.option("-k", "--keywords", help="keywords for query")
@click.option(
    "-r",
    "--region",
    default="wt-wt",
    help="wt-wt, us-en, uk-en, ru-ru, etc. - search region https://duckduckgo.com/params",
)
@click.option(
    "-s",
    "--safesearch",
    default="moderate",
    type=click.Choice(["on", "moderate", "off"]),
    help="Safe Search",
)
@click.option(
    "-t",
    "--time",
    default=None,
    type=click.Choice(["d", "w", "m"]),
    help="search results for the last day, week, month",
)
@click.option(
    "-res", "--resolution", default=None, type=click.Choice(["high", "standart"])
)
@click.option(
    "-d",
    "--duration",
    default=None,
    type=click.Choice(["short", "medium", "long"]),
)
@click.option(
    "-lic",
    "--license_videos",
    default=None,
    type=click.Choice(["creativeCommon", "youtube"]),
)
@click.option(
    "-m",
    "--max_results",
    default=50,
    help="maximum number of results, max=1000, default=50",
)
@click.option(
    "-o",
    "--output",
    default="print",
    help="csv, json (save the results to a csv or json file)",
)
def videos(output, *args, **kwargs):
    data = ddg_videos(output=output, *args, **kwargs)
    if output == "print":
        print_data(data)


@cli.command()
@click.option("-k", "--keywords", help="keywords for query")
@click.option(
    "-r",
    "--region",
    default="wt-wt",
    help="wt-wt, us-en, uk-en, ru-ru, etc. - search region https://duckduckgo.com/params",
)
@click.option(
    "-s",
    "--safesearch",
    default="moderate",
    type=click.Choice(["on", "moderate", "off"]),
    help="Safe Search",
)
@click.option(
    "-t",
    "--time",
    default=None,
    type=click.Choice(["d", "w", "m", "y"]),
    help="d, w, m, y",
)
@click.option(
    "-m",
    "--max_results",
    default=25,
    help="maximum number of results, max=240, default=25",
)
@click.option(
    "-o",
    "--output",
    default="print",
    help="csv, json (save the results to a csv or json file)",
)
def news(output, *args, **kwargs):
    data = ddg_news(output=output, *args, **kwargs)
    if output == "print":
        print_data(data)


@cli.command()
@click.option("-k", "--keywords", help="keywords for query")
@click.option(
    "-p",
    "--place",
    default=None,
    help="simplified search - if set, the other parameters are not used",
)
@click.option("-s", "--street", default=None, help="house number/street")
@click.option("-c", "--city", default=None, help="city of search")
@click.option("-county", "--county", default=None, help="county of search")
@click.option("-state", "--state", default=None, help="state of search")
@click.option("-country", "--country", default=None, help="country of search")
@click.option("-post", "--postalcode", default=None, help="postalcode of search")
@click.option(
    "-lat",
    "--latitude",
    default=None,
    help="""geographic coordinate that specifies the north–south position,
            if latitude and longitude are set, the other parameters are not used""",
)
@click.option(
    "-lon",
    "--longitude",
    default=None,
    help="""geographic coordinate that specifies the east–west position,
            if latitude and longitude are set, the other parameters are not used""",
)
@click.option(
    "-r",
    "--radius",
    default=0,
    help="expand the search square by the distance in kilometers",
)
@click.option("-m", "--max_results", help="number of results, default=None")
@click.option(
    "-o",
    "--output",
    default="print",
    help="csv, json (save the results to a csv or json file)",
)
def maps(output, *args, **kwargs):
    data = ddg_maps(output=output, *args, **kwargs)
    if output == "print":
        print_data(data)


@cli.command()
@click.option("-k", "--keywords", help="text for translation")
@click.option(
    "-f",
    "--from_",
    help="What language to translate from (defaults automatically)",
)
@click.option(
    "-t",
    "--to",
    default="en",
    help="de, ru, fr, etc. What language to translate, defaults='en'",
)
@click.option(
    "-o",
    "--output",
    default="print",
    help="csv, json (save the results to a csv or json file)",
)
def translate(output, *args, **kwargs):
    data = ddg_translate(output=output, *args, **kwargs)
    if output == "print":
        print_data(data)


@cli.command()
@click.option("-k", "--keywords", help="keywords for query")
@click.option(
    "-r",
    "--region",
    default="wt-wt",
    help="wt-wt, us-en, uk-en, ru-ru, etc. - search region https://duckduckgo.com/params",
)
@click.option(
    "-o",
    "--output",
    default="print",
    help="csv, json (save the results to a csv or json file)",
)
def suggestions(output, *args, **kwargs):
    data = ddg_suggestions(output=output, *args, **kwargs)
    if output == "print":
        print_data(data)


if __name__ == "__main__":
    cli(prog_name="ddgs")

# -*- coding: utf-8 -*-
from gtts import gTTS, gTTSError, __version__
from gtts.lang import tts_langs
import click
import logging
import logging.config

# Click settings
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

# Logger settings
LOGGER_SETTINGS = {
    "version": 1,
    "formatters": {"default": {"format": "%(name)s - %(levelname)s - %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "default"}},
    "loggers": {"gtts": {"handlers": ["console"], "level": "WARNING"}},
}

# Logger
logging.config.dictConfig(LOGGER_SETTINGS)
log = logging.getLogger("gtts")


def sys_encoding():
    """Charset to use for --file <path>|- (stdin)"""
    return "utf8"


def validate_text(ctx, param, text):
    """Validation callback for the <text> argument.
    Ensures <text> (arg) and <file> (opt) are mutually exclusive
    """
    if not text and "file" not in ctx.params:
        # No <text> and no <file>
        raise click.BadParameter("<text> or -f/--file <file> required")
    if text and "file" in ctx.params:
        # Both <text> and <file>
        raise click.BadParameter("<text> and -f/--file <file> can't be used together")
    return text


def validate_lang(ctx, param, lang):
    """Validation callback for the <lang> option.
    Ensures <lang> is a supported language unless the <nocheck> flag is set
    """
    if ctx.params["nocheck"]:
        return lang

    try:
        if lang not in tts_langs():
            raise click.UsageError(
                "'%s' not in list of supported languages.\n"
                "Use --all to list languages or "
                "add --nocheck to disable language check." % lang
            )
        else:
            # The language is valid.
            # No need to let gTTS re-validate.
            ctx.params["nocheck"] = True
    except RuntimeError as e:
        # Only case where the <nocheck> flag can be False
        # Non-fatal. gTTS will try to re-validate.
        log.debug(str(e), exc_info=True)

    return lang


def print_languages(ctx, param, value):
    """Callback for <all> flag.
    Prints formatted sorted list of supported languages and exits
    """
    if not value or ctx.resilient_parsing:
        return

    try:
        langs = tts_langs()
        langs_str_list = sorted("{}: {}".format(k, langs[k]) for k in langs)
        click.echo("  " + "\n  ".join(langs_str_list))
    except RuntimeError as e:  # pragma: no cover
        log.debug(str(e), exc_info=True)
        raise click.ClickException("Couldn't fetch language list.")
    ctx.exit()


def set_debug(ctx, param, debug):
    """Callback for <debug> flag.
    Sets logger level to DEBUG
    """
    if debug:
        log.setLevel(logging.DEBUG)
    return


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("text", metavar="<text>", required=False, callback=validate_text)
@click.option(
    "-f",
    "--file",
    metavar="<file>",
    # For py2.7/unicode. If encoding not None Click uses io.open
    type=click.File(encoding=sys_encoding()),
    help="Read from <file> instead of <text>.",
)
@click.option(
    "-o",
    "--output",
    metavar="<file>",
    type=click.File(mode="wb"),
    help="Write to <file> instead of stdout.",
)
@click.option("-s", "--slow", default=False, is_flag=True, help="Read more slowly.")
@click.option(
    "-l",
    "--lang",
    metavar="<lang>",
    default="en",
    show_default=True,
    callback=validate_lang,
    help="IETF language tag. Language to speak in. List documented tags with --all.",
)
@click.option(
    "-t",
    "--tld",
    metavar="<tld>",
    default="com",
    show_default=True,
    is_eager=True,  # Prioritize <tld> to ensure it gets set before <lang>
    help="Top-level domain for the Google host, i.e https://translate.google.<tld>",
)
@click.option(
    "--nocheck",
    default=False,
    is_flag=True,
    is_eager=True,  # Prioritize <nocheck> to ensure it gets set before <lang>
    help="Disable strict IETF language tag checking. Allow undocumented tags.",
)
@click.option(
    "--all",
    default=False,
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=print_languages,
    help="Print all documented available IETF language tags and exit.",
)
@click.option(
    "--debug",
    default=False,
    is_flag=True,
    is_eager=True,  # Prioritize <debug> to see debug logs of callbacks
    expose_value=False,
    callback=set_debug,
    help="Show debug information.",
)
@click.version_option(version=__version__)
def tts_cli(text, file, output, slow, tld, lang, nocheck):
    """Read <text> to mp3 format using Google Translate's Text-to-Speech API
    (set <text> or --file <file> to - for standard input)
    """

    # stdin for <text>
    if text == "-":
        text = click.get_text_stream("stdin").read()

    # stdout (when no <output>)
    if not output:
        output = click.get_binary_stream("stdout")

    # <file> input (stdin on '-' is handled by click.File)
    if file:
        try:
            text = file.read()
        except UnicodeDecodeError as e:  # pragma: no cover
            log.debug(str(e), exc_info=True)
            raise click.FileError(
                file.name, "<file> must be encoded using '%s'." % sys_encoding()
            )

    # TTS
    try:
        tts = gTTS(text=text, lang=lang, slow=slow, tld=tld, lang_check=not nocheck)
        tts.write_to_fp(output)
    except (ValueError, AssertionError) as e:
        raise click.UsageError(str(e))
    except gTTSError as e:
        raise click.ClickException(str(e))

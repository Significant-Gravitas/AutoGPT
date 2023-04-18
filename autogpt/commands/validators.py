from cerberus import Validator

from autogpt.logs import logger

class CommandsValidator:
    command_name: str
    arguments: any

    def __init__(self, command_name, arguments) -> None:
        self.command_name = command_name
        self.arguments = arguments


    def validate_browse_site_command(self) -> None:
        """Validate the browse_site command."""

        url_schema = {'url': {'type': 'string', 'regex': '^(https?|ftp)://[^ \t\n\r\f\v/$.?#].[^ \t\n\r\f\v]*$'}}
        v=Validator(url_schema)
        if not v.validate(self.arguments):
            logger.warn('arguments error: %s', str(v.errors))
            raise Exception("argument url must be a valid url")


    def validate_command(self) -> None:
        """Validate the command."""
        if self.command_name == "browse_site":
            return self.validate_browse_site_command()
        else:
            """skip validation for other commands"""
            logger.debug("skipping validation for command: %s", self.command_name)

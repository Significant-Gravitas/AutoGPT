"""
This module contains the runner for the v2 agent server and client.
"""
import click

from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.IN_PROGRESS,
    handoff_notes=(
        "Before times: A basic client and server is being sketched out. It is at the idea stage.\n"
        "5/9: Had a chat with Mer and David about the design of the message broker and a webservice first approach, which I think is smart.\n"
        "     Sketched out some of the ideas for that. Taking a break for now.\n"
        "5/10: David chat round 2.  Worked out 3 concrete application designs to ground work \n"
        "      1 client <-> 1 agent [existing autogpt]\n"
        "      1 client <-> multi-agent [multi-project autogpt]\n"
        "      multi-client <-> multi-agent [autogpt service]\n"
        "      Did more backend work to mock webserver with message broker.\n"
        "5/11: Async deffed everything so far.\n"
        "      Made it the rest of the way through agent bootstrapping. Need solutions for plugins and memory,\n"
        "      but those can come later by other folks.\n"
        "      Moved agent and agent factory inside context objects to represent the processes they'll run in.\n"
        "      Started working on agent launching.\n"
    ),
)

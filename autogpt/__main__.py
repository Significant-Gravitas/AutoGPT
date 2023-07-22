#!/usr/bin/env python

from autopack.errors import AutoPackError
from autopack.get_pack import try_get_packs
from autopack.installation import install_pack
from autopack.pack import Pack
from autopack.selection import select_packs
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

load_dotenv()


def install_packs(pack_ids: list[str]) -> list[Pack]:
    packs = []
    for pack_id in pack_ids:
        try:
            packs.append(install_pack(pack_id, force_dependencies=True))
        except AutoPackError as e:
            print(f"Pack {pack_id} could not be installed, leaving it out of the toolset. {e}")
            continue

    return packs


def main():
    print("What would you like me to do?")
    print("> ", end="")
    user_input = input()

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")

    pack_ids = select_packs(user_input, llm)
    packs = try_get_packs(pack_ids)

    for pack in packs:
        # TODO: Automatically determine how to pass init args. e.g. map env variables to API key args
        pack.init_tool()

    agent_executor = initialize_agent(
        packs,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    agent_executor.run(user_input)


if __name__ == "__main__":
    main()

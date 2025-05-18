import React from "react";
import SearchHistoryChip from "../SearchHistoryChip";
import IntegrationChip from "../IntegrationChip";
import Block from "../Block";
import { DefaultStateType } from "./BlockMenuDefault";

interface SuggestionContentProps {
  integration: string;
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
  setDefaultState: React.Dispatch<React.SetStateAction<DefaultStateType>>;
}

const SuggestionContent: React.FC<SuggestionContentProps> = ({
  integration,
  setIntegration,
  setDefaultState,
}) => {
  return (
    <div className="scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-zinc-200 scrollbar-track-transparent h-full overflow-y-scroll pt-4">
      <div className="w-full space-y-6 pb-4">
        {/* Recent Searches */}
        <div className="space-y-2.5">
          <p className="px-4 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Recent searches
          </p>
          <div className="scrollbar-hide flex flex-nowrap gap-2 overflow-x-auto">
            <SearchHistoryChip content="image generator" className="ml-4" />
            <SearchHistoryChip content="deepfake" />
            <SearchHistoryChip content="competitor analysis" />
            <SearchHistoryChip content="image generator" />
            <SearchHistoryChip content="deepfake" />
            <SearchHistoryChip content="competitor analysis" />
            <SearchHistoryChip content="image generator" />
            <SearchHistoryChip content="deepfake" />
            <SearchHistoryChip content="competitor analysis" />
          </div>
        </div>

        {/* Integrations */}
        <div className="space-y-2.5 px-4">
          <p className="font-sans text-xs font-medium leading-[1.25rem] text-zinc-500">
            Integrations
          </p>
          <div className="grid grid-cols-3 grid-rows-2 gap-2">
            <IntegrationChip
              icon_url="/integrations/x.png"
              name="Twitter"
              onClick={() => {
                setDefaultState("integrations");
                setIntegration("Twitter Blocks");
              }}
            />
            <IntegrationChip
              icon_url="/integrations/github.png"
              name="Github"
            />
            <IntegrationChip
              icon_url="/integrations/hubspot.png"
              name="Hubspot"
            />
            <IntegrationChip
              icon_url="/integrations/discord.png"
              name="Discord"
            />
            <IntegrationChip
              icon_url="/integrations/medium.png"
              name="Medium"
            />
            <IntegrationChip
              icon_url="/integrations/todoist.png"
              name="Todoist"
            />
          </div>
        </div>

        {/* Top blocks */}
        <div className="space-y-2.5 px-4">
          <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Top blocks
          </p>
          <div className="space-y-2">
            <Block
              title="Find in Dictionary"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Find in Dictionary"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Find in Dictionary"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Find in Dictionary"
              description="Enables your agent to chat with users in natural language."
            />
            <Block
              title="Find in Dictionary"
              description="Enables your agent to chat with users in natural language."
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default SuggestionContent;

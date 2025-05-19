import React from "react";
import { DefaultStateType } from "./BlockMenuDefault";
import SuggestionContent from "./SuggestionContent";
import AllBlocksContent from "./AllBlocksContent";
import IntegrationsContent from "./IntegrationsContent";
import MarketplaceAgentsContent from "./MarketplaceAgentsContent";
import MyAgentsContent from "./MyAgentsContent";
import ActionBlocksContent from "./ActionBlocksContent";
import InputBlocksContent from "./InputBlocksContent";
import OutputBlocksContent from "./OutputBlocksContent";

interface BlockMenuDefaultContentProps {
  defaultState: DefaultStateType;
  setDefaultState: React.Dispatch<React.SetStateAction<DefaultStateType>>;
  integration: string;
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
  setSearchQuery: React.Dispatch<React.SetStateAction<string>>;
}

export interface ActionBlock {
  id: number;
  title: string;
  description: string;
}

export interface BlockListType {
  id: number;
  title: string;
  description: string;
}

const BlockMenuDefaultContent: React.FC<BlockMenuDefaultContentProps> = ({
  defaultState,
  setDefaultState,
  integration,
  setSearchQuery,
  setIntegration,
}) => {
  return (
    <div className="h-full flex-1 overflow-hidden">
      {defaultState == "suggestion" && (
        <SuggestionContent
          setIntegration={setIntegration}
          setSearchQuery={setSearchQuery}
          setDefaultState={setDefaultState}
        />
      )}
      {defaultState == "all_blocks" && <AllBlocksContent />}
      {defaultState == "input_blocks" && <InputBlocksContent />}
      {defaultState == "action_blocks" && <ActionBlocksContent />}
      {defaultState == "output_blocks" && <OutputBlocksContent />}
      {defaultState == "integrations" && (
        <IntegrationsContent
          integration={integration}
          setIntegration={setIntegration}
        />
      )}
      {defaultState == "marketplace_agents" && <MarketplaceAgentsContent />}
      {defaultState == "my_agents" && <MyAgentsContent />}
    </div>
  );
};

export default BlockMenuDefaultContent;

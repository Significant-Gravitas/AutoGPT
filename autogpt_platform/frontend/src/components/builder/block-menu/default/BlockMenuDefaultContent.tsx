import React, { useState } from "react";
import { DefaultStateType } from "./BlockMenuDefault";
import SuggestionContent from "./SuggestionContent";
import AllBlocksContent from "./AllBlocksContent";
import InputBlocksContent from "./InputBlocksContent";
import ActionBlocksContent from "./ActionBlocksContent";
import OutputBlocksContent from "./OutputBlocksContent";
import IntegrationsContent from "./IntegrationsContent";
import MarketplaceAgentsContent from "./MarketplaceAgentsContent";
import MyAgentsContent from "./MyAgentsContent";

interface BlockMenuDefaultContentProps {
  defaultState: DefaultStateType;
  setDefaultState: React.Dispatch<React.SetStateAction<DefaultStateType>>;
  integration: string;
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
}

const BlockMenuDefaultContent: React.FC<BlockMenuDefaultContentProps> = ({
  defaultState,
  setDefaultState,
  integration,
  setIntegration,
}) => {
  return (
    <div className="h-full flex-1 overflow-hidden">
      {defaultState == "suggestion" && (
        <SuggestionContent
          integration={integration}
          setIntegration={setIntegration}
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

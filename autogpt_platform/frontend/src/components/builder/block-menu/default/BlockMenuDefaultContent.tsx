import React from "react";
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
}

const BlockMenuDefaultContent: React.FC<BlockMenuDefaultContentProps> = ({
  defaultState,
}) => {
  return (
    <div className="scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-zinc-200 scrollbar-track-transparent h-full flex-1 overflow-y-scroll pt-4">
      {defaultState == "suggestion" && <SuggestionContent />}
      {defaultState == "all_blocks" && <AllBlocksContent />}
      {defaultState == "input_blocks" && <InputBlocksContent />}
      {defaultState == "action_blocks" && <ActionBlocksContent />}
      {defaultState == "output_blocks" && <OutputBlocksContent />}
      {defaultState == "integrations" && <IntegrationsContent />}
      {defaultState == "marketplace_agents" && <MarketplaceAgentsContent />}
      {defaultState == "my_agents" && <MyAgentsContent />}
    </div>
  );
};

export default BlockMenuDefaultContent;

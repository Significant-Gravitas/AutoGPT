import React from "react";
import { SuggestionContent } from "./SuggestionContent";
import { AllBlocksContent } from "./AllBlocksContent";
import { IntegrationsContent } from "./IntegrationsContent";
import { MarketplaceAgentsContent } from "./MarketplaceAgentsContent";
import { MyAgentsContent } from "./MyAgentsContent";
import { ActionBlocksContent } from "./ActionBlocksContent";
import { InputBlocksContent } from "./InputBlocksContent";
import { OutputBlocksContent } from "./OutputBlocksContent";
import { useBlockMenuContext } from "../block-menu-provider";

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

export const BlockMenuDefaultContent = () => {
  const { defaultState } = useBlockMenuContext();

  return (
    <div className="h-full flex-1 overflow-hidden">
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

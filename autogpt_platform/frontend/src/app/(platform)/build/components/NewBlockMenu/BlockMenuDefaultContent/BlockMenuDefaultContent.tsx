import React from "react";
import { useBlockMenuContext } from "../block-menu-provider";
import { AllBlocksContent } from "../AllBlocksContent/AllBlocksContent";
import { PaginatedBlocksContent } from "../PaginatedBlocksContent/PaginatedBlocksContent";
import { IntegrationsContent } from "../IntegrationsContent/IntegrationsContent";
import { MarketplaceAgentsContent } from "../MarketplaceAgentsContent/MarketplaceAgentsContent";
import { MyAgentsContent } from "../MyAgentsContent/MyAgentsContent";
import { SuggestionContent } from "../SuggestionContent/SuggestionContent";

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
      {defaultState == "input_blocks" && (
        <PaginatedBlocksContent type="input" />
      )}
      {defaultState == "action_blocks" && (
        <PaginatedBlocksContent type="action" />
      )}
      {defaultState == "output_blocks" && (
        <PaginatedBlocksContent type="output" />
      )}
      {defaultState == "integrations" && <IntegrationsContent />}
      {defaultState == "marketplace_agents" && <MarketplaceAgentsContent />}
      {defaultState == "my_agents" && <MyAgentsContent />}
    </div>
  );
};
import React from "react";
import { SuggestionContent } from "./SuggestionContent";
import { AllBlocksContent } from "./AllBlocksContent";
import { IntegrationsContent } from "./IntegrationsContent";
import { MarketplaceAgentsContent } from "./MarketplaceAgentsContent";
import { MyAgentsContent } from "./MyAgentsContent";
import { useBlockMenuContext } from "../block-menu-provider";
import { PaginatedBlocksContent } from "./PaginatedBlocksContent";

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
        <PaginatedBlocksContent blockRequest={{ type: "input" }} />
      )}
      {defaultState == "action_blocks" && (
        <PaginatedBlocksContent blockRequest={{ type: "action" }} />
      )}
      {defaultState == "output_blocks" && (
        <PaginatedBlocksContent blockRequest={{ type: "output" }} />
      )}
      {defaultState == "integrations" && <IntegrationsContent />}
      {defaultState == "marketplace_agents" && <MarketplaceAgentsContent />}
      {defaultState == "my_agents" && <MyAgentsContent />}
    </div>
  );
};

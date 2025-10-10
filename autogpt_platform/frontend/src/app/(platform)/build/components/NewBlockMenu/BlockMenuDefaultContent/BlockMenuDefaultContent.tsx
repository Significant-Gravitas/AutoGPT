import React from "react";
import { AllBlocksContent } from "../AllBlocksContent/AllBlocksContent";
import { PaginatedBlocksContent } from "../PaginatedBlocksContent/PaginatedBlocksContent";
import { IntegrationsContent } from "../IntegrationsContent/IntegrationsContent";
import { MarketplaceAgentsContent } from "../MarketplaceAgentsContent/MarketplaceAgentsContent";
import { MyAgentsContent } from "../MyAgentsContent/MyAgentsContent";
import { SuggestionContent } from "../SuggestionContent/SuggestionContent";
import { useBlockMenuStore } from "../../../stores/blockMenuStore";
import { DefaultStateType } from "../types";

export const BlockMenuDefaultContent = () => {
  const { defaultState } = useBlockMenuStore();

  return (
    <div className="h-full flex-1 overflow-hidden">
      {defaultState == DefaultStateType.SUGGESTION && <SuggestionContent />}
      {defaultState == DefaultStateType.ALL_BLOCKS && <AllBlocksContent />}
      {defaultState == DefaultStateType.INPUT_BLOCKS && (
        <PaginatedBlocksContent type="input" />
      )}
      {defaultState == DefaultStateType.ACTION_BLOCKS && (
        <PaginatedBlocksContent type="action" />
      )}
      {defaultState == DefaultStateType.OUTPUT_BLOCKS && (
        <PaginatedBlocksContent type="output" />
      )}
      {defaultState == DefaultStateType.INTEGRATIONS && <IntegrationsContent />}
      {defaultState == DefaultStateType.MARKETPLACE_AGENTS && (
        <MarketplaceAgentsContent />
      )}
      {defaultState == DefaultStateType.MY_AGENTS && <MyAgentsContent />}
    </div>
  );
};

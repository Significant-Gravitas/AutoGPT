import React from "react";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { IntegrationBlocks } from "../IntegrationBlocks/IntegrationBlocks";
import { PaginatedIntegrationList } from "../PaginatedIntegrationList/PaginatedIntegrationList";
import { cn } from "@/lib/utils";
import { useBlockMenuStore } from "../../../../stores/blockMenuStore";

export const IntegrationsContent = () => {
  const { integration } = useBlockMenuStore();

  if (!integration) {
    return <PaginatedIntegrationList />;
  }

  return (
    <div
      className={cn(
        scrollbarStyles,
        "h-full overflow-y-auto pt-4 transition-all duration-200",
      )}
    >
      <div className="w-full px-4 pb-4">
        <IntegrationBlocks />
      </div>
    </div>
  );
};

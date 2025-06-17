import React from "react";
import { PaginatedIntegrationList } from "./PaginatedIntegrationList";
import { IntegrationBlocks } from "./IntegrationBlocks";
import { useBlockMenuContext } from "../block-menu-provider";

export const IntegrationsContent = () => {
  const { integration } = useBlockMenuContext();

  if (!integration) {
    return <PaginatedIntegrationList />;
  }

  return (
    <div className="scrollbar-thumb-rounded scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200 h-full overflow-y-auto pt-4 transition-all duration-200">
      <div className="w-full px-4 pb-4">
        <IntegrationBlocks />
      </div>
    </div>
  );
};

import React from "react";
import { PaginatedIntegrationList } from "./PaginatedIntegrationList";
import { IntegrationBlocks } from "./IntegrationBlocks";
import { useBlockMenuContext } from "../block-menu-provider";
import { scrollbarStyles } from "@/components/styles/scrollbar";

export const IntegrationsContent = () => {
  const { integration } = useBlockMenuContext();

  if (!integration) {
    return <PaginatedIntegrationList />;
  }

  return (
    <div className={scrollbarStyles}>
      <div className="w-full px-4 pb-4">
        <IntegrationBlocks />
      </div>
    </div>
  );
};

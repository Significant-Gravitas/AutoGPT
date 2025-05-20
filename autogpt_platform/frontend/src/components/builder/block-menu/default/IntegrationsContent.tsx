import React from "react";
import IntegrationList from "./IntegrationList";
import IntegrationBlocks from "./IntegrationBlocks";
import { useBlockMenuContext } from "../block-menu-provider";

const IntegrationsContent: React.FC = () => {
  const { integration } = useBlockMenuContext();
  return (
    <div className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
      <div className="w-full px-4 pb-4">
        {integration == "" ? <IntegrationList /> : <IntegrationBlocks />}
      </div>
    </div>
  );
};

export default IntegrationsContent;

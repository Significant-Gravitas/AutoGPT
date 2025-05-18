import React from "react";
import IntegrationList from "./IntegrationList";
import IntegrationBlocks from "./IntegrationBlocks";

interface IntegrationsContentProps {
  integration: string;
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
}

const IntegrationsContent: React.FC<IntegrationsContentProps> = ({
  integration,
  setIntegration,
}) => {
  // I am currently comparing the integration with names, in future maybe using ids
  return (
    <div className="scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-zinc-200 scrollbar-track-transparent h-full overflow-y-scroll pt-4">
      <div className="w-full px-4 pb-4">
        {integration == "" ? (
          <IntegrationList setIntegration={setIntegration} />
        ) : (
          <IntegrationBlocks
            integration={integration}
            setIntegration={setIntegration}
          />
        )}
      </div>
    </div>
  );
};

export default IntegrationsContent;

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
  return (
    <div className="scrollbar-thumb-rounded h-full overflow-y-scroll pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
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

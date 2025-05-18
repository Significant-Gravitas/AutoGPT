import React from "react";
import IntegrationBlock from "../IntegrationBlock";
import Integration from "../Integration";
interface IntegrationListProps {
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
}

const IntegrationList: React.FC<IntegrationListProps> = ({
  setIntegration,
}) => {
  return (
    <div className="space-y-3">
      <Integration
        title="Twitter Blocks"
        icon_url="/integrations/x.png"
        description="All twitter blocks, It has everthing to interact with twitter"
        number_of_blocks={10}
        onClick={() => setIntegration("Twitter Blocks")}
      />
      <Integration
        title="Discord Blocks"
        icon_url="/integrations/discord.png"
        description="All Discord blocks, It has everthing to interact with discord"
        number_of_blocks={14}
        onClick={() => setIntegration("Discord Blocks")}
      />
      <Integration
        title="Github Blocks"
        icon_url="/integrations/github.png"
        description="All Github blocks, It has everthing to interact with github"
        number_of_blocks={4}
        onClick={() => setIntegration("Github Blocks")}
      />
      <Integration
        title="Hubspot Blocks"
        icon_url="/integrations/hubspot.png"
        description="All Hubspot blocks, It has everthing to interact with Hubspot"
        number_of_blocks={2}
        onClick={() => setIntegration("Hubspot Blocks")}
      />
    </div>
  );
};

export default IntegrationList;

import React from "react";
import MarketplaceAgentBlock from "../MarketplaceAgentBlock";

const MarketplaceAgentsContent: React.FC = () => {
  return (
    <div className="scrollbar-thin scrollbar-thumb-rounded scrollbar-thumb-zinc-200 scrollbar-track-transparent h-full overflow-y-scroll pt-4">
      <div className="w-full space-y-3 px-4 pb-4">
        <MarketplaceAgentBlock
          title="turtle test"
          image_url="/placeholder.png"
          creator_name="Autogpt"
          number_of_runs={1000}
        />
        <MarketplaceAgentBlock
          title="turtle test 1"
          image_url="/placeholder.png"
          creator_name="Autogpt"
          number_of_runs={1324}
        />
        <MarketplaceAgentBlock
          title="turtle test 2"
          image_url="/placeholder.png"
          creator_name="Autogpt"
          number_of_runs={10030}
        />
        <MarketplaceAgentBlock
          title="turtle test 3"
          image_url="/placeholder.png"
          creator_name="Autogpt"
          number_of_runs={324}
        />
        <MarketplaceAgentBlock
          title="turtle test"
          image_url="/placeholder.png"
          creator_name="Autogpt"
          number_of_runs={4345}
        />
        <MarketplaceAgentBlock
          title="turtle test"
          image_url="/placeholder.png"
          creator_name="Autogpt"
          number_of_runs={324}
        />
        <MarketplaceAgentBlock
          title="turtle test 3"
          image_url="/placeholder.png"
          creator_name="Autogpt"
          number_of_runs={324}
        />
        <MarketplaceAgentBlock
          title="turtle test"
          image_url="/placeholder.png"
          creator_name="Autogpt"
          number_of_runs={4345}
        />
        <MarketplaceAgentBlock
          title="turtle test"
          image_url="/placeholder.png"
          creator_name="Autogpt"
          number_of_runs={324}
        />
      </div>
    </div>
  );
};

export default MarketplaceAgentsContent;

import { Button } from "@/components/ui/button";
import React from "react";
import IntegrationBlock from "../IntegrationBlock";

interface IntegrationBlocksProps {
  integration: string;
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
}

const IntegrationBlocks: React.FC<IntegrationBlocksProps> = ({
  integration,
  setIntegration,
}) => {
  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Button
            variant={"link"}
            className="p-0 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800"
            onClick={() => {
              setIntegration("");
            }}
          >
            Integrations
          </Button>
          <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            /
          </p>
          <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            {integration}
          </p>
        </div>
        <span className="flex h-[1.375rem] w-[1.6875rem] items-center justify-center rounded-[1.25rem] bg-[#f0f0f0] p-1.5 font-sans text-sm leading-[1.375rem] text-zinc-500 group-disabled:text-zinc-400">
          {13}
        </span>
      </div>
      {integration == "Twitter Blocks" && (
        <div className="space-y-3">
          <IntegrationBlock
            title={`${integration}: Post tweet`}
            description="Post tweet on twitter"
            icon_url="/integrations/x.png"
          />
          <IntegrationBlock
            title={`${integration}: Delete tweet`}
            description="Delete tweet on twitter"
            icon_url="/integrations/x.png"
          />
          <IntegrationBlock
            title={`${integration}: Update tweet`}
            description="Update tweet on twitter"
            icon_url="/integrations/x.png"
          />
          <IntegrationBlock
            title={`${integration}: Retweet tweet`}
            description="Retweet tweet on twitter"
            icon_url="/integrations/x.png"
          />
        </div>
      )}

      {integration == "Discord Blocks" && (
        <div className="space-y-3">
          <IntegrationBlock
            title={`${integration}: Create`}
            description="Create message on discord"
            icon_url="/integrations/discord.png"
          />
          <IntegrationBlock
            title={`${integration}: Delete`}
            description="Delete message on discord"
            icon_url="/integrations/discord.png"
          />
          <IntegrationBlock
            title={`${integration}: Update`}
            description="Update message on discord"
            icon_url="/integrations/discord.png"
          />
          <IntegrationBlock
            title={`${integration}: Read`}
            description="Read message on discord"
            icon_url="/integrations/discord.png"
          />
        </div>
      )}
    </div>
  );
};

export default IntegrationBlocks;

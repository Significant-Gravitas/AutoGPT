import { Button } from "@/components/ui/button";
import React, { useState, useEffect } from "react";
import IntegrationBlock from "../IntegrationBlock";
import {
  integrationBlocksData,
  integrationsListData,
} from "../../testing_data";

export interface IntegrationBlockData {
  title: string;
  description: string;
  icon_url: string;
}

interface IntegrationBlocksProps {
  integration: string;
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
}

const IntegrationBlocks: React.FC<IntegrationBlocksProps> = ({
  integration,
  setIntegration,
}) => {
  const [blocks, setBlocks] = useState<IntegrationBlockData[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // TEMPORARY FETCHING
  useEffect(() => {
    if (integration) {
      setIsLoading(true);
      setTimeout(() => {
        const foundBlocks = integrationBlocksData[integration] || [];
        setBlocks(foundBlocks);
        setIsLoading(false);
      }, 800);
    }
  }, [integration]);

  const getBlockCount = (): number => {
    const integrationData = integrationsListData.find(
      (item) => item.title === integration,
    );
    return integrationData?.number_of_blocks || 0;
  };

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
          {getBlockCount()}
        </span>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {Array(5)
            .fill(0)
            .map((_, index) => (
              <IntegrationBlock.Skeleton key={index} />
            ))}
        </div>
      ) : (
        <div className="space-y-3">
          {blocks.map((block, index) => (
            <IntegrationBlock
              key={index}
              title={block.title}
              description={block.description}
              icon_url={block.icon_url}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default IntegrationBlocks;

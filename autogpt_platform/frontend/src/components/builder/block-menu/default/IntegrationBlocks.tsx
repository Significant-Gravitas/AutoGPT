import { Button } from "@/components/ui/button";
import React, { useState, useEffect } from "react";
import IntegrationBlock from "../IntegrationBlock";
import { useBlockMenuContext } from "../block-menu-provider";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { Block } from "@/lib/autogpt-server-api";

const IntegrationBlocks: React.FC = ({}) => {
  const { integration, setIntegration, addNode } = useBlockMenuContext();
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const api = useBackendAPI();

  useEffect(() => {
    const fetchBlocks = async () => {
      if (integration) {
        setIsLoading(true);
        const response = await api.getBuilderBlocks({ provider: integration });
        setBlocks(response.blocks);
        setIsLoading(false);
      }
    };

    fetchBlocks();
  }, [api, integration]);

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Button
            variant={"link"}
            className="p-0 font-sans text-sm font-medium leading-[1.375rem] text-zinc-800"
            onClick={() => {
              setIntegration(null);
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
          {blocks.length}
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
              title={block.name}
              description={block.description}
              icon_url={`/integrations/${integration}.png`}
              onClick={() => {
                addNode(
                  block.id,
                  block.name,
                  block.hardcodedValues || {},
                  block,
                );
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default IntegrationBlocks;

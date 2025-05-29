import { Button } from "@/components/ui/button";
import React, { useState, useEffect, Fragment } from "react";
import IntegrationBlock from "../IntegrationBlock";
import { useBlockMenuContext } from "../block-menu-provider";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { Block } from "@/lib/autogpt-server-api";
import ErrorState from "../ErrorState";
import { Skeleton } from "@/components/ui/skeleton";

const IntegrationBlocks: React.FC = ({}) => {
  const { integration, setIntegration, addNode } = useBlockMenuContext();
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const api = useBackendAPI();

  const fetchBlocks = async () => {
    if (integration) {
      try {
        setLoading(true);
        setError(null);
        const response = await api.getBuilderBlocks({ provider: integration });
        setBlocks(response.blocks);
      } catch (err) {
        console.error("Failed to fetch integration blocks:", err);
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load integration blocks",
        );
      } finally {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    fetchBlocks();
  }, [api, integration]);

  if (loading) {
    return (
      <div className="w-full space-y-3 p-4">
        {[0, 1, 3].map((blockIndex) => (
          <Fragment key={blockIndex}>
            {blockIndex > 0 && (
              <Skeleton className="my-4 h-[1px] w-full text-zinc-100" />
            )}
            {[0, 1, 2].map((index) => (
              <IntegrationBlock.Skeleton key={`${blockIndex}-${index}`} />
            ))}
          </Fragment>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full p-4">
        <ErrorState
          title="Failed to load integration blocks"
          error={error}
          onRetry={fetchBlocks}
        />
      </div>
    );
  }

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
      <div className="space-y-3">
        {blocks.map((block, index) => (
          <IntegrationBlock
            key={index}
            title={block.name}
            description={block.description}
            icon_url={`/integrations/${integration}.png`}
            onClick={() => {
              addNode(block);
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default IntegrationBlocks;

import React from "react";
import { IntegrationChip } from "../IntegrationChip";
import { Block } from "../Block";
import { useBlockMenuContext } from "../block-menu-provider";
import { useSuggestionContent } from "./useSuggestionContent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { blockMenuContainerStyle } from "../style";

export const SuggestionContent = () => {
  const { setIntegration, setDefaultState } = useBlockMenuContext();
  const { suggestions, isLoading, isError, error, refetch } = useSuggestionContent();

  if (isError) {
    return (
      <div className="h-full p-4">
        <ErrorCard
          isSuccess={false}
          responseError={{detail: error?.detail || "Error fetching suggestions"}}
          context="Error fetching suggestions"
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  return (
    <div className={blockMenuContainerStyle}>
      <div className="w-full space-y-6 pb-4">
        {/* Integrations */}
        <div className="space-y-2.5 px-4">
          <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Integrations
          </p>
          <div className="grid grid-cols-3 grid-rows-2 gap-2">
            {!isLoading && suggestions
              ? suggestions.providers.map((provider, index) => (
                  <IntegrationChip
                    key={`integration-${index}`}
                    icon_url={`/integrations/${provider}.png`}
                    name={provider}
                    onClick={() => {
                      setDefaultState("integrations");
                      setIntegration(provider);
                    }}
                  />
                ))
              : Array(6)
                  .fill(0)
                  .map((_, index) => (
                    <IntegrationChip.Skeleton
                      key={`integration-skeleton-${index}`}
                    />
                  ))}
          </div>
        </div>

        {/* Top blocks */}
        <div className="space-y-2.5 px-4">
          <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Top blocks
          </p>
          <div className="space-y-2">
            {!isLoading && suggestions
              ? suggestions.top_blocks.map((block, index) => (
                  <Block
                    key={`block-${index}`}
                    title={block.name}
                    description={block.description}
                  />
                ))
              : Array(3)
                  .fill(0)
                  .map((_, index) => (
                    <Block.Skeleton key={`block-skeleton-${index}`} />
                  ))}
          </div>
        </div>
      </div>
    </div>
  );
};

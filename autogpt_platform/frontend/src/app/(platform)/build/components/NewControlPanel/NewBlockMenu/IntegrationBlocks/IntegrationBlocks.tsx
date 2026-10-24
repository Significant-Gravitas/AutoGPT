import { Button } from "@/components/__legacy__/ui/button";
import React, { Fragment } from "react";
import { IntegrationBlock } from "../IntergrationBlock";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { useIntegrationBlocks } from "./useIntegrationBlocks";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { useBlockMenuStore } from "../../../../stores/blockMenuStore";
import { useShallow } from "zustand/react/shallow";
import { Text } from "@/components/atoms/Text/Text";

export const IntegrationBlocks = () => {
  const { integration, setIntegration } = useBlockMenuStore(
    useShallow((state) => ({
      integration: state.integration,
      setIntegration: state.setIntegration,
    })),
  );
  const {
    allBlocks,
    status,
    totalBlocks,
    blocksLoading,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    error,
    refetch,
  } = useIntegrationBlocks();

  if (blocksLoading) {
    return (
      <div className="w-full space-y-3 p-4">
        {Array.from({ length: 3 }).map((_, blockIndex) => (
          <Fragment key={blockIndex}>
            {blockIndex > 0 && (
              <Skeleton
                // TODO(tokens): no fitting role for text-zinc-100 #EFEFF0; #EFEFF0 is surface-hover (a surface fill), separator (a border colour)
                className="my-4 h-[1px] w-full text-zinc-100"
              />
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
        <ErrorCard
          isSuccess={false}
          responseError={error || undefined}
          httpError={{
            status: status,
            statusText: "Request failed",
            message: (error?.detail as string) || "An error occurred",
          }}
          context="block menu"
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  return (
    <InfiniteScroll
      isFetchingNextPage={isFetchingNextPage}
      fetchNextPage={fetchNextPage}
      hasNextPage={hasNextPage}
    >
      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Button
              variant={"link"}
              className="p-0 font-sans text-sm font-medium leading-5.5 text-foreground-emphasis"
              onClick={() => {
                setIntegration(undefined);
              }}
            >
              Integrations
            </Button>
            <Text variant="body-medium" tone="emphasis">
              /
            </Text>
            <Text variant="body-medium" tone="emphasis">
              {integration}
            </Text>
          </div>
          <span
            // TODO(tokens): no exact role: bg-[#f0f0f0] #F0F0F0
            className="flex h-[1.375rem] w-[1.6875rem] items-center justify-center rounded-[1.25rem] bg-[#f0f0f0] p-1.5 font-sans text-sm leading-5.5 text-foreground-tertiary group-disabled:text-foreground-subtle"
          >
            {totalBlocks}
          </span>
        </div>
        <div className="space-y-3">
          {allBlocks.map((block) => (
            <IntegrationBlock
              key={block.id}
              title={block.name}
              description={block.description}
              blockData={block}
              icon_url={`/integrations/${integration}.png`}
            />
          ))}
        </div>
      </div>
    </InfiniteScroll>
  );
};

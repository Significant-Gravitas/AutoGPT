import { Button } from "@/components/atoms/Button/Button";
import React, { Fragment } from "react";
import { IntegrationBlock } from "../IntergrationBlock";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { useIntegrationBlocks } from "./useIntegrationBlocks";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { useBlockMenuStore } from "../../../../stores/blockMenuStore";

export const IntegrationBlocks = () => {
  const { integration, setIntegration } = useBlockMenuStore();
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
              <Skeleton className="my-4 h-px w-full text-zinc-100" />
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
              variant="link"
              className="h-auto min-w-0 p-0 font-sans text-sm leading-5.5 font-medium text-zinc-800 no-underline underline-offset-4 hover:underline"
              onClick={() => {
                setIntegration(undefined);
              }}
            >
              Integrations
            </Button>
            <Text variant="body-medium" className="text-zinc-800">
              /
            </Text>
            <Text variant="body-medium" className="text-zinc-800">
              {integration}
            </Text>
          </div>
          <span className="flex h-5.5 w-6.75 items-center justify-center rounded-2xl bg-zinc-100 p-1.5 font-sans text-sm leading-5.5 text-zinc-500 group-disabled:text-zinc-400">
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

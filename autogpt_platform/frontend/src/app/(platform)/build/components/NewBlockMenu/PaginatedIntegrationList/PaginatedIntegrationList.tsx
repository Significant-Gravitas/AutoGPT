import React from "react";
import { Integration } from "../Integration";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { usePaginatedIntegrationList } from "./usePaginatedIntegrationList";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { blockMenuContainerStyle } from "../style";
import { useBlockMenuStore } from "../../../stores/blockMenuStore";

export const PaginatedIntegrationList = () => {
  const { setIntegration } = useBlockMenuStore();
  const {
    allProviders: providers,
    providersLoading,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    error,
    status,
    refetch,
  } = usePaginatedIntegrationList();

  if (error) {
    return (
      <div className="h-full px-4">
        <ErrorCard
          isSuccess={false}
          responseError={error || undefined}
          context="block menu"
          httpError={{
            status: status,
            statusText: "Request failed",
            message: (error?.detail as string) || "An error occurred",
          }}
          onRetry={() => refetch()}
        />
      </div>
    );
  }
  if (providersLoading && providers.length === 0) {
    return (
      <div className={blockMenuContainerStyle}>
        {Array.from({ length: 6 }).map((_, integrationIndex) => (
          <Integration.Skeleton key={integrationIndex} />
        ))}
      </div>
    );
  }

  return (
    <InfiniteScroll
      isFetchingNextPage={isFetchingNextPage}
      fetchNextPage={fetchNextPage}
      hasNextPage={hasNextPage}
      className={blockMenuContainerStyle}
    >
      {providers.map((integration, index) => (
        <Integration
          key={integration.name + index}
          title={integration.name}
          icon_url={`/integrations/${integration.name}.png`}
          description={integration.description}
          number_of_blocks={integration.integration_count}
          onClick={() => setIntegration(integration.name)}
        />
      ))}
    </InfiniteScroll>
  );
};

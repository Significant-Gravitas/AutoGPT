import React from "react";
import { Integration } from "../Integration";
import { useBlockMenuContext } from "../block-menu-provider";
import { usePagination } from "@/hooks/usePagination";
import { ErrorState } from "../ErrorState";

export const PaginatedIntegrationList = () => {
  const { setIntegration } = useBlockMenuContext();
  const {
    data: providers,
    loading,
    loadingMore,
    hasMore,
    error,
    scrollRef,
    refresh,
  } = usePagination({
    request: { apiType: "providers" },
    pageSize: 10,
  });

  if (loading) {
    return (
      <div
        ref={scrollRef}
        className="scrollbar-thumb-rounded scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200 h-full overflow-y-auto pt-4 transition-all duration-200"
      >
        <div className="w-full space-y-3 px-4 pb-4">
          {Array.from({ length: 6 }).map((_, integrationIndex) => (
            <Integration.Skeleton key={integrationIndex} />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full p-4">
        <ErrorState
          title="Failed to load integrations"
          error={error}
          onRetry={refresh}
        />
      </div>
    );
  }

  return (
    <div
      ref={scrollRef}
      className="scrollbar-thumb-rounded scrollbar-thin scrollbar-track-transparent scrollbar-thumb-transparent hover:scrollbar-thumb-zinc-200 h-full overflow-y-auto pt-4 transition-all duration-200"
    >
      <div className="w-full px-4 pb-4">
        <div className="space-y-3">
          {providers.map((integration, index) => (
            <Integration
              key={index}
              title={integration.name}
              icon_url={`/integrations/${integration.name}.png`}
              description={integration.description}
              number_of_blocks={integration.integration_count}
              onClick={() => setIntegration(integration.name)}
            />
          ))}
          {loadingMore && hasMore && (
            <>
              {Array.from({ length: 3 }).map((_, index) => (
                <Integration.Skeleton key={`loading-${index}`} />
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

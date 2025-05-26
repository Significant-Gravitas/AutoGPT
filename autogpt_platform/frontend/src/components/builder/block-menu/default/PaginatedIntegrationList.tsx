import React from "react";
import Integration from "../Integration";
import { Button } from "@/components/ui/button";
import { useBlockMenuContext } from "../block-menu-provider";
import { usePagination } from "@/hooks/usePagination";

const PaginatedIntegrationList: React.FC = () => {
  const { setIntegration } = useBlockMenuContext();
  const { data: providers, loading, loadingMore, hasMore, error, scrollRef, refresh } = usePagination({
    request: { apiType: 'providers' },
    pageSize: 10,
  });

  return (
    <div 
      ref={scrollRef}
      className="scrollbar-thumb-rounded h-full overflow-y-auto pt-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200"
    >
      <div className="w-full px-4 pb-4">
        <div className="space-y-3">
          {loading
            ? Array(5)
                .fill(null)
                .map((_, index) => (
                  <Integration.Skeleton key={index} />
                ))
            : providers.map((integration, index) => (
                <Integration
                  key={index}
                  title={integration.name}
                  icon_url={`/integrations/${integration.name}.png`}
                  description={integration.description}
                  number_of_blocks={integration.integration_count}
                  onClick={() => setIntegration(integration.name)}
                />
              ))}
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-3">
              <p className="text-sm text-red-600 mb-2">
                Error loading integrations: {error}
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={refresh}
                className="h-7 text-xs"
              >
                Retry
              </Button>
            </div>
          )}
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

export default PaginatedIntegrationList;
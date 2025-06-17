import React, { useCallback, useEffect, useState } from "react";
import { IntegrationChip } from "../IntegrationChip";
import { Block } from "../Block";
import { useBlockMenuContext } from "../block-menu-provider";
import {
  CredentialsProviderName,
  SuggestionsResponse,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { ErrorState } from "../ErrorState";
import { scrollbarStyles } from "@/components/styles/scrollbar";

export const SuggestionContent = () => {
  const { setIntegration, setDefaultState, addNode } = useBlockMenuContext();

  const [suggestionsData, setSuggestionsData] =
    useState<SuggestionsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const api = useBackendAPI();

  const fetchSuggestions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await api.getSuggestions();
      setSuggestionsData(response);
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(
        err instanceof Error ? err.message : "Failed to load suggestions",
      );
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    fetchSuggestions();
  }, [fetchSuggestions]);

  if (error) {
    return (
      <div className="h-full p-4">
        <ErrorState
          title="Failed to load suggestions"
          error={error}
          onRetry={fetchSuggestions}
        />
      </div>
    );
  }

  return (
    <div className={scrollbarStyles}>
      <div className="w-full space-y-6 pb-4">
        {/* Integrations */}
        <div className="space-y-2.5 px-4">
          <p className="font-sans text-sm font-medium leading-[1.375rem] text-zinc-800">
            Integrations
          </p>
          <div className="grid grid-cols-3 grid-rows-2 gap-2">
            {!loading && suggestionsData
              ? suggestionsData.providers.map((provider, index) => (
                  <IntegrationChip
                    key={`integration-${index}`}
                    icon_url={`/integrations/${provider}.png`}
                    name={provider}
                    onClick={() => {
                      setDefaultState("integrations");
                      setIntegration(provider as CredentialsProviderName);
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
            {!loading && suggestionsData
              ? suggestionsData.top_blocks.map((block, index) => (
                  <Block
                    key={`block-${index}`}
                    title={block.name}
                    description={block.description}
                    onClick={() => {
                      addNode(block);
                    }}
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

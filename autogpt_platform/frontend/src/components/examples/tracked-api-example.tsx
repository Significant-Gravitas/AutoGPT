"use client";

import { useState } from "react";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useAPIWithTracking } from "@/hooks/use-api-with-tracking";
import { TrackedButton } from "@/components/atoms/Button";
import { EventKeys } from "@/services/feature-flags/use-track-event";

/**
 * Example component showing how to use API tracking
 * This pattern can be applied to any component that makes API calls
 */
export function TrackedAPIExample() {
  const api = useBackendAPI();
  const { trackAPICall, trackValidationError } = useAPIWithTracking();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreateAgent = async () => {
    setLoading(true);
    setError(null);

    try {
      // Validate input first
      const agentName = "My New Agent";
      if (!agentName || agentName.length < 3) {
        trackValidationError(
          "agentName",
          "Agent name must be at least 3 characters",
          "agent-creation-form",
        );
        setError("Agent name must be at least 3 characters");
        return;
      }

      // Make API call with tracking
      const agent = await trackAPICall(
        () =>
          api.createGraph({
            name: agentName,
            description: "Created with tracking",
          } as any),
        "create-agent",
        {
          agentName,
          source: "example-component",
        },
      );

      console.log("Agent created successfully:", agent);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create agent");
    } finally {
      setLoading(false);
    }
  };

  const handleRunAgent = async (agentId: string) => {
    setLoading(true);
    setError(null);

    try {
      // Track API call with performance metrics
      const execution = await trackAPICall(
        () =>
          api.executeGraph(agentId as any, {
            inputData: {},
          } as any),
        "run-agent",
        {
          agentId,
          source: "example-component",
        },
      );

      console.log("Agent execution started:", execution);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run agent");
    } finally {
      setLoading(false);
    }
  };

  const handleFetchMarketplace = async () => {
    setLoading(true);
    setError(null);

    try {
      // Example of tracking a marketplace API call
      const storeItems = await trackAPICall(
        () => (api as any).getStoreListings({ limit: 10 } as any),
        "fetch-marketplace",
        {
          limit: 10,
          source: "example-component",
        },
      );

      console.log("Marketplace items fetched:", storeItems);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to fetch marketplace",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 p-4">
      <h2 className="text-xl font-bold">API Tracking Examples</h2>

      {error && (
        <div className="rounded-md bg-red-50 p-3 text-red-700">{error}</div>
      )}

      <div className="flex gap-2">
        <TrackedButton
          onClick={handleCreateAgent}
          loading={loading}
          trackEventKey={EventKeys.AGENT_CREATED}
          trackEventData={{ source: "api-example" }}
        >
          Create Agent (with API tracking)
        </TrackedButton>

        <TrackedButton
          onClick={() => handleRunAgent("test-agent-id")}
          loading={loading}
          variant="secondary"
          trackEventKey={EventKeys.AGENT_RUN_STARTED}
          trackEventData={{ source: "api-example" }}
        >
          Run Agent (with API tracking)
        </TrackedButton>

        <TrackedButton
          onClick={handleFetchMarketplace}
          loading={loading}
          variant="outline"
          trackEventKey={EventKeys.STORE_ACCESSED}
          trackEventData={{ source: "api-example" }}
        >
          Fetch Marketplace (with API tracking)
        </TrackedButton>
      </div>

      <div className="mt-4 rounded-md bg-gray-100 p-3">
        <p className="text-sm text-gray-600">
          This component demonstrates how to use:
        </p>
        <ul className="mt-2 list-inside list-disc text-sm text-gray-600">
          <li>TrackedButton for UI interaction tracking</li>
          <li>trackAPICall for API performance and error tracking</li>
          <li>trackValidationError for form validation tracking</li>
          <li>Automatic error reporting to LaunchDarkly</li>
        </ul>
      </div>
    </div>
  );
}

import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetV2GetMyAgentsQueryOptions,
  getGetV2GetMyLiveSubmissionQueryOptions,
} from "@/app/api/__generated__/endpoints/store/store";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { okData } from "@/app/api/helpers";
import { type AgentFormSource, toAgentFormSource } from "./helpers";

type CheckStatus = "idle" | "checking" | "failed";

/**
 * Asks for a workflow's live Marketplace listing before the dialog builds its
 * form, so a slow or failed answer never opens the new-listing form for a
 * listed workflow. A check the dialog has moved on from (closed, or started
 * another) is dropped when it lands.
 */
export function useLiveListingCheck() {
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<CheckStatus>("idle");
  const latestCheck = useRef(0);

  async function run<T>(
    load: () => Promise<T>,
    onChecked: (result: T) => void,
  ) {
    const check = ++latestCheck.current;
    setStatus("checking");
    try {
      const result = await load();
      if (check !== latestCheck.current) return;
      setStatus("idle");
      onChecked(result);
    } catch {
      if (check === latestCheck.current) setStatus("failed");
    }
  }

  async function fetchLiveSubmission(graphId: string) {
    const response = await queryClient.fetchQuery(
      getGetV2GetMyLiveSubmissionQueryOptions(graphId, {
        query: { staleTime: 0 },
      }),
    );
    if (response.status !== 200) {
      throw new Error(`Live listing lookup answered ${response.status}`);
    }
    return response.data;
  }

  async function fetchAgent(graphId: string) {
    const response = await queryClient.fetchQuery(
      getGetV2GetMyAgentsQueryOptions(undefined, { query: { staleTime: 0 } }),
    );
    const agents = okData(response)?.agents;
    return toAgentFormSource(agents?.find((a) => a.graph_id === graphId));
  }

  function checkPickedAgent(
    graphId: string,
    onChecked: (live: StoreSubmission | null) => void,
  ) {
    void run(() => fetchLiveSubmission(graphId), onChecked);
  }

  function checkPreSelectedAgent(
    graphId: string,
    onChecked: (live: StoreSubmission | null, agent: AgentFormSource) => void,
  ) {
    void run(
      () => Promise.all([fetchLiveSubmission(graphId), fetchAgent(graphId)]),
      ([live, agent]) => onChecked(live, agent),
    );
  }

  function cancel() {
    latestCheck.current += 1;
    setStatus("idle");
  }

  return {
    checkPickedAgent,
    checkPreSelectedAgent,
    cancel,
    isChecking: status === "checking",
    hasFailed: status === "failed",
  };
}

import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { computeInitialAgentInputs } from "./helpers";
import { InputValues } from "./types";
import {
  useGetV2GetAgentByVersion,
  useGetV2GetAgentGraph,
} from "@/app/api/__generated__/endpoints/store/store";
import { resolveResponse } from "@/app/api/helpers";
import { postV2AddMarketplaceAgent } from "@/app/api/__generated__/endpoints/library/library";
import { GraphID } from "@/lib/autogpt-server-api";

export function useOnboardingRunStep() {
  const onboarding = useOnboarding(undefined, "AGENT_CHOICE");

  const [showInput, setShowInput] = useState(false);
  const [agent, setAgent] = useState<GraphMeta | null>(null);
  const [storeAgent, setStoreAgent] = useState<StoreAgentDetails | null>(null);
  const [runningAgent, setRunningAgent] = useState(false);

  const [inputCredentials, setInputCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >({});

  const [credentialsValid, setCredentialsValid] = useState(true);
  const [credentialsLoaded, setCredentialsLoaded] = useState(false);

  const { toast } = useToast();
  const router = useRouter();
  const api = useBackendAPI();

  const currentAgentVersion =
    onboarding.state?.selectedStoreListingVersionId ?? "";

  const storeAgentQuery = useGetV2GetAgentByVersion(currentAgentVersion, {
    query: { enabled: !!currentAgentVersion },
  });

  const graphMetaQuery = useGetV2GetAgentGraph(currentAgentVersion, {
    query: { enabled: !!currentAgentVersion },
  });

  useEffect(() => {
    onboarding.setStep(5);
  }, []);

  useEffect(() => {
    if (storeAgentQuery.data && storeAgentQuery.data.status === 200) {
      setStoreAgent(storeAgentQuery.data.data);
    }
  }, [storeAgentQuery.data]);

  useEffect(() => {
    if (
      graphMetaQuery.data &&
      graphMetaQuery.data.status === 200 &&
      onboarding.state
    ) {
      const graphMeta = graphMetaQuery.data.data as GraphMeta;

      setAgent(graphMeta);

      const update = computeInitialAgentInputs(
        graphMeta,
        (onboarding.state.agentInput as unknown as InputValues) || null,
      );

      onboarding.updateState({ agentInput: update });
    }
  }, [graphMetaQuery.data]);

  function handleNewRun() {
    if (!onboarding.state) return;

    setShowInput(true);
    onboarding.setStep(6);
    onboarding.completeStep("AGENT_NEW_RUN");
  }

  function handleSetAgentInput(key: string, value: string) {
    if (!onboarding.state) return;

    onboarding.updateState({
      agentInput: {
        ...onboarding.state.agentInput,
        [key]: value,
      },
    });
  }

  async function handleRunAgent() {
    if (!agent || !storeAgent || !onboarding.state) {
      toast({
        title: "Error getting agent",
        description:
          "Either the agent is not available or there was an error getting it.",
        variant: "destructive",
      });

      return;
    }

    setRunningAgent(true);

    try {
      const libraryAgent = await resolveResponse(
        postV2AddMarketplaceAgent({
          store_listing_version_id: storeAgent?.store_listing_version_id || "",
          source: "onboarding",
        }),
      );

      const { id: runID } = await api.executeGraph(
        libraryAgent.graph_id as GraphID,
        libraryAgent.graph_version,
        onboarding.state.agentInput || {},
        inputCredentials,
        "onboarding",
      );

      onboarding.updateState({ onboardingAgentExecutionId: runID });

      router.push("/onboarding/6-congrats");
    } catch (error) {
      console.error("Error running agent:", error);

      toast({
        title: "Error running agent",
        description:
          "There was an error running your agent. Please try again or try choosing a different agent if it still fails.",
        variant: "destructive",
      });

      setRunningAgent(false);
    }
  }

  return {
    ready: graphMetaQuery.isSuccess && storeAgentQuery.isSuccess,
    error: graphMetaQuery.error || storeAgentQuery.error,
    agent,
    onboarding,
    showInput,
    storeAgent,
    runningAgent,
    credentialsValid,
    credentialsLoaded,
    handleSetAgentInput,
    handleRunAgent,
    handleNewRun,
    handleCredentialsChange: setInputCredentials,
    handleCredentialsValidationChange: setCredentialsValid,
    handleCredentialsLoadingChange: (v: boolean) => setCredentialsLoaded(!v),
  };
}

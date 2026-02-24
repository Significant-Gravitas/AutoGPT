import { useToast } from "@/components/molecules/Toast/use-toast";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { computeInitialAgentInputs } from "./helpers";
import { InputValues } from "./types";
import { okData, resolveResponse } from "@/app/api/helpers";
import { postV2AddMarketplaceAgent } from "@/app/api/__generated__/endpoints/library/library";
import {
  useGetV2GetAgentByVersion,
  useGetV2GetAgentGraph,
} from "@/app/api/__generated__/endpoints/store/store";
import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { GraphID } from "@/lib/autogpt-server-api";

export function useOnboardingRunStep() {
  const onboarding = useOnboarding(undefined, "AGENT_CHOICE");

  const [showInput, setShowInput] = useState(false);
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

  const {
    data: storeAgent,
    error: storeAgentQueryError,
    isSuccess: storeAgentQueryIsSuccess,
  } = useGetV2GetAgentByVersion(currentAgentVersion, {
    query: {
      enabled: !!currentAgentVersion,
      select: okData,
    },
  });

  const {
    data: agentGraphMeta,
    error: agentGraphQueryError,
    isSuccess: agentGraphQueryIsSuccess,
  } = useGetV2GetAgentGraph(currentAgentVersion, {
    query: {
      enabled: !!currentAgentVersion,
      select: okData,
    },
  });

  useEffect(() => {
    onboarding.setStep(5);
  }, []);

  useEffect(() => {
    if (agentGraphMeta && onboarding.state) {
      const initialAgentInputs = computeInitialAgentInputs(
        agentGraphMeta,
        (onboarding.state.agentInput as unknown as InputValues) || null,
      );

      onboarding.updateState({ agentInput: initialAgentInputs });
    }
  }, [agentGraphMeta]);

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
    if (!agentGraphMeta || !storeAgent || !onboarding.state) {
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
    ready: agentGraphQueryIsSuccess && storeAgentQueryIsSuccess,
    error: agentGraphQueryError || storeAgentQueryError,
    agentGraph: agentGraphMeta || null,
    onboarding,
    showInput,
    storeAgent: storeAgent || null,
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

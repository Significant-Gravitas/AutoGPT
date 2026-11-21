import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { useCallback, useEffect, useState } from "react";
import { PublishAgentInfoInitialData } from "./components/AgentInfoStep/helpers";
import { useRouter } from "next/navigation";
import {
  buildInitialFormData,
  emptyModalState,
  findPublishedSubmission,
} from "./helpers";
import {
  useGetV2GetMyAgents,
  useGetV2GetUserProfile,
  useGetV2ListMySubmissions,
  getGetV2ListMySubmissionsQueryKey,
} from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import type { MyUnpublishedAgent } from "@/app/api/__generated__/models/myUnpublishedAgent";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/lib/auth/hooks/useAuth";

const defaultTargetState: PublishState = {
  isOpen: false,
  step: "select",
  submissionData: null,
};

export type PublishStep = "select" | "info" | "review";

export type PublishState = {
  isOpen: boolean;
  step: PublishStep;
  submissionData: StoreSubmission | null;
};

export interface Props {
  trigger?: React.ReactNode;
  targetState?: PublishState;
  onStateChange?: (state: PublishState) => void;
  onRequestEdit?: (submission: StoreSubmission) => void;
  preSelectedAgentId?: string;
  preSelectedAgentVersion?: number;
  showTrigger?: boolean;
}

export function usePublishAgentModal({
  targetState,
  onStateChange,
  preSelectedAgentId,
  preSelectedAgentVersion,
}: Props) {
  const [currentState, setCurrentState] = useState<PublishState>(
    targetState || defaultTargetState,
  );

  const updateState = useCallback(
    (newState: PublishState) => {
      setCurrentState(newState);
      onStateChange?.(newState);
    },
    [onStateChange],
  );

  const [initialData, setInitialData] =
    useState<PublishAgentInfoInitialData>(emptyModalState);

  const [_, setSelectedAgent] = useState<string | null>(null);

  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(
    preSelectedAgentId || null,
  );

  const [selectedAgentVersion, setSelectedAgentVersion] = useState<
    number | null
  >(preSelectedAgentVersion || null);

  const router = useRouter();
  const queryClient = useQueryClient();
  const { isLoggedIn } = useAuth();

  // Fetch agent data for pre-populating form when agent is pre-selected
  const { data: myAgents } = useGetV2GetMyAgents(undefined, {
    query: {
      enabled: isLoggedIn,
    },
  });
  const { data: mySubmissions } = useGetV2ListMySubmissions(undefined, {
    query: {
      enabled: isLoggedIn,
    },
  });
  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: (x) => x.data as ProfileDetails,
      enabled: isLoggedIn,
    },
  });
  const creatorUsername = profile?.username;

  // Sync currentState with targetState when it changes from outside
  useEffect(() => {
    if (targetState) {
      setCurrentState(targetState);
    }
  }, [targetState]);

  // Reset internal state when modal opens (only on initial open, not on every targetState change)
  const [hasOpened, setHasOpened] = useState(false);
  useEffect(() => {
    if (!targetState) return;
    if (targetState.isOpen && !hasOpened) {
      setSelectedAgent(null);
      setSelectedAgentId(preSelectedAgentId || null);
      setSelectedAgentVersion(preSelectedAgentVersion || null);
      setInitialData(emptyModalState);
      setHasOpened(true);
    } else if (!targetState.isOpen && hasOpened) {
      setHasOpened(false);
    }
  }, [targetState, preSelectedAgentId, preSelectedAgentVersion]);

  // Pre-populate form data when modal opens with info step and pre-selected agent
  useEffect(() => {
    if (
      !targetState?.isOpen ||
      targetState.step !== "info" ||
      !preSelectedAgentId ||
      !preSelectedAgentVersion
    )
      return;
    const agentsData = okData(myAgents);
    const submissionsData = okData(mySubmissions);

    if (!agentsData || !submissionsData) return;

    // Find the agent data
    const agent = agentsData.agents?.find(
      (a: MyUnpublishedAgent) => a.graph_id === preSelectedAgentId,
    );
    if (!agent) return;

    // Find published submission data for this agent (for updates)
    const publishedSubmissionData = findPublishedSubmission(
      submissionsData.submissions,
      preSelectedAgentId,
    );

    const initialFormData = buildInitialFormData(
      preSelectedAgentId,
      {
        name: agent.agent_name,
        description: agent.description || "",
        imageSrc: agent.agent_image || "",
        recommendedScheduleCron: agent.recommended_schedule_cron || null,
      },
      publishedSubmissionData,
    );

    setInitialData(initialFormData);

    // Update the state with the submission data if this is an update
    if (publishedSubmissionData) {
      setCurrentState((prevState) => ({
        ...prevState,
        submissionData: publishedSubmissionData,
      }));
    }
  }, [
    targetState,
    preSelectedAgentId,
    preSelectedAgentVersion,
    myAgents,
    mySubmissions,
  ]);

  function handleClose() {
    // Reset all internal state
    setSelectedAgent(null);
    setSelectedAgentId(null);
    setSelectedAgentVersion(null);
    setInitialData(emptyModalState);

    // Invalidate submissions query to refresh the data after modal closes
    queryClient.invalidateQueries({
      queryKey: getGetV2ListMySubmissionsQueryKey(),
    });

    // Update parent with clean closed state
    const newState = {
      isOpen: false,
      step: "select" as PublishStep,
      submissionData: null,
    };
    updateState(newState);
  }

  function handleAgentSelect(agentName: string) {
    setSelectedAgent(agentName);
  }

  function handleNextFromSelect(
    agentId: string,
    agentVersion: number,
    agentData: {
      name: string;
      description: string;
      imageSrc: string;
      recommendedScheduleCron: string | null;
    },
  ) {
    // The picker passes no submission. The backend files a submission for an
    // already-listed graph as its listing's next version, so open that update.
    const publishedSubmissionData = findPublishedSubmission(
      okData(mySubmissions)?.submissions,
      agentId,
    );

    const initialFormData = buildInitialFormData(
      agentId,
      agentData,
      publishedSubmissionData,
    );

    setInitialData(initialFormData);

    updateState({
      ...currentState,
      step: "info",
      submissionData: publishedSubmissionData || null,
    });

    setSelectedAgentId(agentId);
    setSelectedAgentVersion(agentVersion);
  }

  function handleSuccessFromInfo(submissionData: any) {
    updateState({
      ...currentState,
      submissionData: submissionData,
      step: "review",
    });
  }

  function handleBack() {
    if (currentState.step === "info") {
      // When the modal was opened pre-scoped to a specific agent (e.g. from
      // the builder or library), there is no upstream picker to go back to —
      // close the modal instead of surfacing a picker the caller intentionally
      // skipped.
      if (preSelectedAgentId) {
        handleClose();
        return;
      }
      updateState({
        ...currentState,
        step: "select",
      });
    } else if (currentState.step === "review") {
      updateState({
        ...currentState,
        step: "info",
      });
    }
  }

  function handleGoToDashboard() {
    router.push("/settings/creator-dashboard");
    handleClose();
  }

  function handleGoToBuilder() {
    router.push("/build");
    handleClose();
  }

  return {
    // handlers
    handleClose,
    handleAgentSelect,
    handleNextFromSelect,
    handleGoToDashboard,
    handleGoToBuilder,
    handleSuccessFromInfo,
    handleBack,
    // state
    currentState,
    updateState,
    initialData,
    selectedAgentId,
    selectedAgentVersion,
    creatorUsername,
  };
}

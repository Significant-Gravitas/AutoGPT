import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { useCallback, useEffect, useState } from "react";
import { PublishAgentInfoInitialData } from "./components/AgentInfoStep/helpers";
import { useRouter } from "next/navigation";
import { emptyModalState } from "./helpers";
import {
  useGetV2GetMyAgents,
  useGetV2ListMySubmissions,
  getGetV2ListMySubmissionsQueryKey,
} from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import type { MyAgent } from "@/app/api/__generated__/models/myAgent";
import { useQueryClient } from "@tanstack/react-query";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

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
  const { isLoggedIn } = useSupabase();

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
    const agentsData = okData(myAgents) as any;
    const submissionsData = okData(mySubmissions) as any;

    if (!agentsData || !submissionsData) return;

    // Find the agent data
    const agent = agentsData.agents?.find(
      (a: MyAgent) => a.agent_id === preSelectedAgentId,
    );
    if (!agent) return;

    // Find published submission data for this agent (for updates)
    const publishedSubmissionData = submissionsData.submissions
      ?.filter(
        (s: StoreSubmission) =>
          s.status === "APPROVED" && s.agent_id === preSelectedAgentId,
      )
      .sort(
        (a: StoreSubmission, b: StoreSubmission) =>
          b.agent_version - a.agent_version,
      )[0];

    // Populate initial data (same logic as handleNextFromSelect)
    const initialFormData: PublishAgentInfoInitialData = publishedSubmissionData
      ? {
          agent_id: preSelectedAgentId,
          title: publishedSubmissionData.name,
          subheader: publishedSubmissionData.sub_heading || "",
          description: publishedSubmissionData.description,
          instructions: publishedSubmissionData.instructions || "",
          youtubeLink: publishedSubmissionData.video_url || "",
          agentOutputDemo: publishedSubmissionData.agent_output_demo_url || "",
          additionalImages: [
            ...new Set(publishedSubmissionData.image_urls || []),
          ].filter(Boolean) as string[],
          category: publishedSubmissionData.categories?.[0] || "",
          thumbnailSrc: agent.agent_image || "https://picsum.photos/300/200",
          slug: publishedSubmissionData.slug,
          recommendedScheduleCron: agent.recommended_schedule_cron || "",
          changesSummary: publishedSubmissionData.changes_summary || "",
        }
      : {
          ...emptyModalState,
          agent_id: preSelectedAgentId,
          title: agent.agent_name,
          description: agent.description || "",
          thumbnailSrc: agent.agent_image || "https://picsum.photos/300/200",
          slug: agent.agent_name.replace(/ /g, "-"),
          recommendedScheduleCron: agent.recommended_schedule_cron || "",
        };

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
    publishedSubmissionData?: StoreSubmission | null,
  ) {
    // Pre-populate with published data if this is an update, otherwise use agent data
    const initialFormData: PublishAgentInfoInitialData = publishedSubmissionData
      ? {
          agent_id: agentId,
          title: publishedSubmissionData.name,
          subheader: publishedSubmissionData.sub_heading || "",
          description: publishedSubmissionData.description,
          instructions: publishedSubmissionData.instructions || "",
          youtubeLink: publishedSubmissionData.video_url || "",
          agentOutputDemo: publishedSubmissionData.agent_output_demo_url || "",
          additionalImages: [
            ...new Set(publishedSubmissionData.image_urls || []),
          ].filter(Boolean) as string[],
          category: publishedSubmissionData.categories?.[0] || "", // Take first category
          thumbnailSrc: agentData.imageSrc, // Use current agent image
          slug: publishedSubmissionData.slug,
          recommendedScheduleCron: agentData.recommendedScheduleCron || "",
          changesSummary: publishedSubmissionData.changes_summary || "", // Pre-populate with existing changes summary
        }
      : {
          ...emptyModalState,
          agent_id: agentId,
          title: agentData.name,
          description: agentData.description,
          thumbnailSrc: agentData.imageSrc,
          slug: agentData.name.replace(/ /g, "-"),
          recommendedScheduleCron: agentData.recommendedScheduleCron || "",
        };

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
    router.push("/profile/dashboard");
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
  };
}

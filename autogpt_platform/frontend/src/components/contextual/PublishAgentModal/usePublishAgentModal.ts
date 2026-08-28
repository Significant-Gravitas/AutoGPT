import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { useCallback, useEffect, useState } from "react";
import { PublishAgentInfoInitialData } from "./components/AgentInfoStep/helpers";
import { useRouter } from "next/navigation";
import { emptyModalState } from "./helpers";
import {
  useGetV2GetMyAgents,
  useGetV2GetUserProfile,
  useGetV2ListMySubmissions,
  getGetV2GetMyAgentsQueryKey,
  getGetV2ListMySubmissionsQueryKey,
} from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";
import type { MyUnpublishedAgent } from "@/app/api/__generated__/models/myUnpublishedAgent";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";

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
  preSelectedOrganizationId?: string | null;
  preSelectedTeamId?: string | null;
  showTrigger?: boolean;
}

export function usePublishAgentModal({
  targetState,
  onStateChange,
  preSelectedAgentId,
  preSelectedAgentVersion,
  preSelectedOrganizationId,
  preSelectedTeamId,
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

  const [selectedAgentOrganizationId, setSelectedAgentOrganizationId] =
    useState<string | null>(preSelectedOrganizationId ?? null);
  const [selectedAgentTeamId, setSelectedAgentTeamId] = useState<string | null>(
    preSelectedTeamId ?? null,
  );

  const router = useRouter();
  const queryClient = useQueryClient();
  const { isLoggedIn } = useAuth();
  const activeOrgID = useOrgTeamStore((state) => state.activeOrgID);
  const activeTeamID = useOrgTeamStore((state) => state.activeTeamID);
  const isTenantReady = useOrgTeamStore((state) => state.isLoaded);
  const queryOrganizationId =
    preSelectedAgentId && preSelectedOrganizationId !== undefined
      ? preSelectedOrganizationId
      : activeOrgID;
  const queryTeamId =
    preSelectedAgentId && preSelectedTeamId !== undefined
      ? preSelectedTeamId
      : activeTeamID;
  const queryRequest = getTenantRequestInit(
    queryOrganizationId,
    queryTeamId,
    isTenantReady,
  );

  // Fetch agent data for pre-populating form when agent is pre-selected
  const { data: myAgents } = useGetV2GetMyAgents(undefined, {
    query: {
      enabled: isLoggedIn && isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getGetV2GetMyAgentsQueryKey(),
        queryOrganizationId,
        queryTeamId,
      ),
    },
    request: queryRequest,
  });
  const { data: mySubmissions } = useGetV2ListMySubmissions(undefined, {
    query: {
      enabled: isLoggedIn && isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getGetV2ListMySubmissionsQueryKey(),
        queryOrganizationId,
        queryTeamId,
      ),
    },
    request: queryRequest,
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
      setSelectedAgentOrganizationId(queryOrganizationId);
      setSelectedAgentTeamId(queryTeamId);
      setInitialData(emptyModalState);
      setHasOpened(true);
    } else if (!targetState.isOpen && hasOpened) {
      setHasOpened(false);
    }
  }, [
    targetState,
    preSelectedAgentId,
    preSelectedAgentVersion,
    queryOrganizationId,
    queryTeamId,
  ]);

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
      (a: MyUnpublishedAgent) =>
        a.graph_id === preSelectedAgentId &&
        (a.organization_id ?? null) === queryOrganizationId &&
        (a.team_id ?? null) === queryTeamId,
    );
    if (!agent) return;

    // Find published submission data for this agent (for updates)
    const publishedSubmissionData = submissionsData.submissions
      ?.filter(
        (s: StoreSubmission) =>
          s.status === "APPROVED" &&
          s.graph_id === preSelectedAgentId &&
          (s.organization_id ?? null) === queryOrganizationId &&
          (s.team_id ?? null) === queryTeamId,
      )
      .sort(
        (a: StoreSubmission, b: StoreSubmission) =>
          b.graph_version - a.graph_version,
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
          thumbnailSrc: agent.agent_image || "",
          slug: publishedSubmissionData.slug,
          recommendedScheduleCron: agent.recommended_schedule_cron || "",
          changesSummary: publishedSubmissionData.changes_summary || "",
        }
      : {
          ...emptyModalState,
          agent_id: preSelectedAgentId,
          title: agent.agent_name,
          description: agent.description || "",
          thumbnailSrc: agent.agent_image || "",
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
    queryOrganizationId,
    queryTeamId,
  ]);

  function handleClose() {
    // Reset all internal state
    setSelectedAgent(null);
    setSelectedAgentId(null);
    setSelectedAgentVersion(null);
    setSelectedAgentOrganizationId(null);
    setSelectedAgentTeamId(null);
    setInitialData(emptyModalState);

    // Invalidate submissions query to refresh the data after modal closes
    queryClient.invalidateQueries({
      queryKey: getTeamScopedQueryKey(
        getGetV2ListMySubmissionsQueryKey(),
        queryOrganizationId,
        queryTeamId,
      ),
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
    organizationId: string | null,
    teamId: string | null,
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
    setSelectedAgentOrganizationId(organizationId);
    setSelectedAgentTeamId(teamId);
  }

  function handleSuccessFromInfo(submissionData: StoreSubmission) {
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
    selectedAgentOrganizationId,
    selectedAgentTeamId,
    creatorUsername,
  };
}

import type { LiveStoreSubmission } from "@/app/api/__generated__/models/liveStoreSubmission";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { useCallback, useEffect, useState } from "react";
import { PublishAgentInfoInitialData } from "./components/AgentInfoStep/helpers";
import { useRouter } from "next/navigation";
import {
  type AgentFormSource,
  buildInitialFormData,
  emptyModalState,
} from "./helpers";
import {
  useGetV2GetUserProfile,
  getGetV2ListMySubmissionsQueryKey,
} from "@/app/api/__generated__/endpoints/store/store";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import { useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useLiveListingCheck } from "./useLiveListingCheck";

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

  // The selected workflow's live listing, as the last check found it. Kept
  // out of currentState, which the page's targetState replaces whenever it
  // changes the dialog's open state, step or submission.
  const [liveSubmission, setLiveSubmission] =
    useState<LiveStoreSubmission | null>(null);
  const listingCheck = useLiveListingCheck();

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

  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: (x) => x.data as ProfileDetails,
      enabled: isLoggedIn,
    },
  });
  const creatorUsername = profile?.username;

  // Follow targetState's values, not its identity: a page may hand in a new
  // object on every render, and the dialog must stay on the step it's on. So
  // re-sending the same values doesn't reset or reopen it; a page that keeps
  // the dialog mounted mirrors onStateChange into what it passes. `target`
  // gives the dependency lists plain values.
  const hasTarget = targetState !== undefined;
  const target = targetState ?? defaultTargetState;
  useEffect(() => {
    if (!hasTarget) return;
    setCurrentState({
      isOpen: target.isOpen,
      step: target.step,
      submissionData: target.submissionData,
    });
  }, [hasTarget, target.isOpen, target.step, target.submissionData]);

  // Reset internal state when modal opens (only on initial open, not on every targetState change)
  const [hasOpened, setHasOpened] = useState(false);
  useEffect(() => {
    if (!hasTarget) return;
    if (target.isOpen && !hasOpened) {
      setSelectedAgent(null);
      setSelectedAgentId(preSelectedAgentId || null);
      setSelectedAgentVersion(preSelectedAgentVersion || null);
      setInitialData(emptyModalState);
      setLiveSubmission(null);
      setHasOpened(true);
      if (target.step === "info") checkPreSelectedAgent();
    } else if (!target.isOpen && hasOpened) {
      // The page closed the dialog: a check still running would reopen it.
      listingCheck.cancel();
      setHasOpened(false);
    }
  }, [
    hasTarget,
    target.isOpen,
    target.step,
    preSelectedAgentId,
    preSelectedAgentVersion,
  ]);

  // A dialog opened pre-scoped to a workflow starts on the listing form; build
  // it once the workflow's live listing is known.
  function checkPreSelectedAgent() {
    if (!preSelectedAgentId) return;
    const agentId = preSelectedAgentId;
    listingCheck.checkPreSelectedAgent(
      agentId,
      preSelectedAgentVersion,
      (live, agent) => {
        setLiveSubmission(live);
        setInitialData(buildInitialFormData(agentId, agent, live ?? undefined));
      },
    );
  }

  function handleClose() {
    // Reset all internal state
    listingCheck.cancel();
    setSelectedAgent(null);
    setSelectedAgentId(null);
    setSelectedAgentVersion(null);
    setInitialData(emptyModalState);
    setLiveSubmission(null);

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
    // A check still running is for the workflow picked before this one.
    listingCheck.cancel();
    setSelectedAgent(agentName);
  }

  function handleNextFromSelect(
    agentId: string,
    agentVersion: number,
    agentData: AgentFormSource,
  ) {
    // The backend files a submission for an already-listed workflow as its
    // listing's next version, so ask for the live one before opening the form.
    listingCheck.checkPickedAgent(agentId, (live) => {
      setLiveSubmission(live);
      setInitialData(
        buildInitialFormData(agentId, agentData, live ?? undefined),
      );
      updateState({ ...currentState, step: "info", submissionData: null });
      setSelectedAgentId(agentId);
      setSelectedAgentVersion(agentVersion);
    });
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
    handleRetryListingCheck: checkPreSelectedAgent,
    // state
    currentState,
    updateState,
    initialData,
    selectedAgentId,
    selectedAgentVersion,
    creatorUsername,
    isMarketplaceUpdate: !!liveSubmission,
    // A pre-scoped dialog has no form to show until its check answers.
    isInfoFormReady: !!initialData.agent_id || !preSelectedAgentId,
    isCheckingListing: listingCheck.isChecking,
    listingCheckFailed: listingCheck.hasFailed,
  };
}

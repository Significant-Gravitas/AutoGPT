import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { useCallback, useEffect, useState } from "react";
import { PublishAgentInfoInitialData } from "./components/AgentInfoStep/helpers";
import { useRouter } from "next/navigation";
import { emptyModalState } from "./helpers";

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
}

export function usePublishAgentModal({ targetState, onStateChange }: Props) {
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

  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);

  const [selectedAgentVersion, setSelectedAgentVersion] = useState<
    number | null
  >(null);

  const router = useRouter();

  // Sync currentState with targetState when it changes from outside
  useEffect(() => {
    if (targetState) {
      setCurrentState(targetState);
    }
  }, [targetState]);

  // Reset internal state when modal opens
  useEffect(() => {
    if (!targetState) return;
    if (targetState.isOpen && targetState.step === "select") {
      setSelectedAgent(null);
      setSelectedAgentId(null);
      setSelectedAgentVersion(null);
      setInitialData(emptyModalState);
    }
  }, [targetState]);

  function handleClose() {
    // Reset all internal state
    setSelectedAgent(null);
    setSelectedAgentId(null);
    setSelectedAgentVersion(null);
    setInitialData(emptyModalState);

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
    setInitialData({
      ...emptyModalState,
      agent_id: agentId,
      title: agentData.name,
      description: agentData.description,
      thumbnailSrc: agentData.imageSrc,
      slug: agentData.name.replace(/ /g, "-"),
      recommendedScheduleCron: agentData.recommendedScheduleCron || "",
    });

    updateState({
      ...currentState,
      step: "info",
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

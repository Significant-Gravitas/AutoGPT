import { getGetV2ListMySubmissionsQueryKey } from "@/app/api/__generated__/endpoints/store/store";
import { StoreSubmissionRequest } from "@/app/api/__generated__/models/storeSubmissionRequest";
import { useCallback, useEffect, useState } from "react";
import { PublishAgentInfoInitialData } from "./components/AgentInfoStep/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToast } from "@/components/molecules/Toast/use-toast";

const defaultTargetState: PublishState = {
  isOpen: false,
  step: "select",
  submissionData: null,
};

export type PublishStep = "select" | "info" | "review";

export type PublishState = {
  isOpen: boolean;
  step: PublishStep;
  submissionData: StoreSubmissionRequest | null;
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

  const [initialData, setInitialData] = useState<PublishAgentInfoInitialData>({
    agent_id: "",
    title: "",
    subheader: "",
    slug: "",
    thumbnailSrc: "",
    youtubeLink: "",
    category: "",
    description: "",
  });

  const [_, setSelectedAgent] = useState<string | null>(null);

  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);

  const [selectedAgentVersion, setSelectedAgentVersion] = useState<
    number | null
  >(null);

  const queryClient = useQueryClient();
  const router = useRouter();
  const api = useBackendAPI();
  const { toast } = useToast();

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
      setInitialData({
        agent_id: "",
        title: "",
        subheader: "",
        slug: "",
        thumbnailSrc: "",
        youtubeLink: "",
        category: "",
        description: "",
      });
    }
  }, [targetState]);

  function handleClose() {
    // Reset all internal state
    setSelectedAgent(null);
    setSelectedAgentId(null);
    setSelectedAgentVersion(null);
    setInitialData({
      agent_id: "",
      title: "",
      subheader: "",
      slug: "",
      thumbnailSrc: "",
      youtubeLink: "",
      category: "",
      description: "",
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
    agentData: { name: string; description: string; imageSrc: string },
  ) {
    setInitialData({
      agent_id: agentId,
      title: agentData.name,
      subheader: "",
      description: agentData.description,
      thumbnailSrc: agentData.imageSrc,
      youtubeLink: "",
      category: "",
      slug: agentData.name.replace(/ /g, "-"),
      additionalImages: [],
    });

    updateState({
      ...currentState,
      step: "info",
    });

    setSelectedAgentId(agentId);
    setSelectedAgentVersion(agentVersion);
  }

  async function handleNextFromInfo(
    name: string,
    subHeading: string,
    slug: string,
    description: string,
    imageUrls: string[],
    videoUrl: string,
    categories: string[],
  ) {
    const missingFields: string[] = [];

    if (!name) missingFields.push("Name");
    if (!subHeading) missingFields.push("Sub-heading");
    if (!description) missingFields.push("Description");
    if (!imageUrls.length) missingFields.push("Image");
    if (!categories.filter(Boolean).length) missingFields.push("Categories");

    if (missingFields.length > 0) {
      toast({
        title: "Missing Required Fields",
        description: `Please fill in: ${missingFields.join(", ")}`,
        duration: 3000,
      });
      return;
    }

    const filteredCategories = categories.filter(Boolean);

    updateState({
      ...currentState,
      submissionData: {
        name,
        sub_heading: subHeading,
        description,
        image_urls: imageUrls,
        video_url: videoUrl,
        agent_id: selectedAgentId || "",
        agent_version: selectedAgentVersion || 0,
        slug,
        categories: filteredCategories,
      },
    });

    // Create store submission
    try {
      const response = await api.createStoreSubmission({
        name: name,
        sub_heading: subHeading,
        description: description,
        image_urls: imageUrls,
        video_url: videoUrl,
        agent_id: selectedAgentId || "",
        agent_version: selectedAgentVersion || 0,
        slug: slug.replace(/\s+/g, "-"),
        categories: filteredCategories,
      });

      updateState({
        ...currentState,
        submissionData: response,
        step: "review",
      });

      await queryClient.invalidateQueries({
        queryKey: getGetV2ListMySubmissionsQueryKey(),
      });
    } catch (error) {
      console.error("Error creating store submission:", error);
    }
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
    handleNextFromInfo,
    handleBack,
    // state
    currentState,
    updateState,
    initialData,
  };
}

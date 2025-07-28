"use client";

import * as React from "react";
import { PublishAgentSelect } from "../../agptui/PublishAgentSelect";
import {
  PublishAgentInfo,
  PublishAgentInfoInitialData,
} from "./components/PublishAgentSelectInfo";
import { PublishAgentAwaitingReview } from "./components/PublishAgentAwaitingReview";
import { MyAgentsResponse } from "@/lib/autogpt-server-api";
import { useRouter } from "next/navigation";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { StoreSubmissionRequest } from "@/app/api/__generated__/models/storeSubmissionRequest";
import { useQueryClient } from "@tanstack/react-query";
import { getGetV2ListMySubmissionsQueryKey } from "@/app/api/__generated__/endpoints/store/store";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/atoms/Button/Button";

export type PublishStep = "select" | "info" | "review";

type PublishState = {
  isOpen: boolean;
  step: PublishStep;
  submissionData: StoreSubmissionRequest | null;
};

interface Props {
  trigger?: React.ReactNode;
  targetState: PublishState;
  onStateChange?: (state: PublishState) => void;
}

export function PublishAgentModal({
  trigger,
  targetState,
  onStateChange,
}: Props) {
  const [currentState, setCurrentState] =
    React.useState<PublishState>(targetState);

  const updateState = React.useCallback(
    (newState: PublishState) => {
      setCurrentState(newState);
      onStateChange?.(newState);
    },
    [onStateChange],
  );

  const [initialData, setInitialData] =
    React.useState<PublishAgentInfoInitialData>({
      agent_id: "",
      title: "",
      subheader: "",
      slug: "",
      thumbnailSrc: "",
      youtubeLink: "",
      category: "",
      description: "",
    });

  const [myAgents, setMyAgents] = React.useState<MyAgentsResponse | null>(null);

  const [_, setSelectedAgent] = React.useState<string | null>(null);

  const [selectedAgentId, setSelectedAgentId] = React.useState<string | null>(
    null,
  );

  const [selectedAgentVersion, setSelectedAgentVersion] = React.useState<
    number | null
  >(null);

  const queryClient = useQueryClient();
  const router = useRouter();
  const api = useBackendAPI();

  const { toast } = useToast();

  React.useEffect(() => {
    setCurrentState(targetState);
    // Reset internal state when modal opens
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

  React.useEffect(() => {
    if (currentState.isOpen) {
      const loadMyAgents = async () => {
        try {
          const response = await api.getMyAgents();
          setMyAgents(response);
        } catch (error) {
          console.error("Failed to load my agents:", error);
        }
      };

      loadMyAgents();
    }
  }, [currentState, api]);

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

  function handleNextFromSelect(agentId: string, agentVersion: number) {
    const selectedAgentData = myAgents?.agents.find(
      (agent) => agent.agent_id === agentId,
    );

    const name = selectedAgentData?.agent_name || "";
    const description = selectedAgentData?.description || "";

    setInitialData({
      agent_id: agentId,
      title: name,
      subheader: "",
      description: description,
      thumbnailSrc: selectedAgentData?.agent_image || "",
      youtubeLink: "",
      category: "",
      slug: name.replace(/ /g, "-"),
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
      await api.createStoreSubmission({
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
      await queryClient.invalidateQueries({
        queryKey: getGetV2ListMySubmissionsQueryKey(),
      });
    } catch (error) {
      console.error("Error creating store submission:", error);
    }

    updateState({
      ...currentState,
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

  function renderContent() {
    switch (currentState.step) {
      case "select":
        return (
          <PublishAgentSelect
            agents={
              myAgents?.agents
                .map((agent) => ({
                  name: agent.agent_name,
                  id: agent.agent_id,
                  version: agent.agent_version,
                  lastEdited: agent.last_edited,
                  imageSrc:
                    agent.agent_image || "https://picsum.photos/300/200",
                }))
                .sort(
                  (a, b) =>
                    new Date(b.lastEdited).getTime() -
                    new Date(a.lastEdited).getTime(),
                ) || []
            }
            onSelect={handleAgentSelect}
            onCancel={handleClose}
            onNext={handleNextFromSelect}
            onClose={handleClose}
            onOpenBuilder={() => router.push("/build")}
          />
        );
      case "info":
        return (
          <PublishAgentInfo
            onBack={handleBack}
            onSubmit={handleNextFromInfo}
            onClose={handleClose}
            initialData={initialData}
          />
        );
      case "review":
        return currentState.submissionData &&
          currentState.submissionData.name ? (
          <PublishAgentAwaitingReview
            agentName={currentState.submissionData.name}
            subheader={currentState.submissionData.sub_heading}
            description={currentState.submissionData.description || ""}
            thumbnailSrc={currentState.submissionData.image_urls?.[0]}
            onClose={handleClose}
            onDone={handleClose}
            onViewProgress={() => {
              router.push("/profile/dashboard");
              handleClose();
            }}
          />
        ) : (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
          </div>
        );
    }
  }

  return (
    <>
      <Dialog
        controlled={{
          isOpen: currentState.isOpen,
          set: (isOpen) => {
            if (!isOpen) {
              // When closing, always reset to clean state
              handleClose();
            } else {
              updateState({
                ...currentState,
                isOpen: isOpen,
              });
            }
          },
        }}
      >
        <Dialog.Trigger>
          {trigger || <Button size="small">Publish Agent</Button>}
        </Dialog.Trigger>
        <Dialog.Content>{renderContent()}</Dialog.Content>
      </Dialog>
    </>
  );
}

"use client";

import * as React from "react";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { PublishAgentSelect } from "../PublishAgentSelect";
import {
  PublishAgentInfo,
  PublishAgentInfoInitialData,
} from "../PublishAgentSelectInfo";
import { PublishAgentAwaitingReview } from "../PublishAgentAwaitingReview";
import { Button } from "../Button";
import {
  StoreSubmissionRequest,
  MyAgentsResponse,
} from "@/lib/autogpt-server-api";
import { useRouter } from "next/navigation";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToast } from "@/components/ui/use-toast";
import AutogptButton from "../AutogptButton";

interface PublishAgentPopoutProps {
  trigger?: React.ReactNode;
  openPopout?: boolean;
  inputStep?: "select" | "info" | "review";
  submissionData?: StoreSubmissionRequest;
}

export const PublishAgentPopout: React.FC<PublishAgentPopoutProps> = ({
  trigger,
  openPopout = false,
  inputStep = "select",
  submissionData = {
    name: "",
    sub_heading: "",
    slug: "",
    description: "",
    image_urls: [],
    agent_id: "",
    agent_version: 0,
    categories: [],
  },
}) => {
  const [step, setStep] = React.useState<"select" | "info" | "review">(
    inputStep,
  );
  const [myAgents, setMyAgents] = React.useState<MyAgentsResponse | null>(null);
  const [selectedAgent, setSelectedAgent] = React.useState<string | null>(null);
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
  const [publishData, setPublishData] =
    React.useState<StoreSubmissionRequest>(submissionData);
  const [selectedAgentId, setSelectedAgentId] = React.useState<string | null>(
    null,
  );
  const [selectedAgentVersion, setSelectedAgentVersion] = React.useState<
    number | null
  >(null);
  const [open, setOpen] = React.useState(false);

  const router = useRouter();
  const api = useBackendAPI();

  const { toast } = useToast();

  React.useEffect(() => {
    setOpen(openPopout);
    setStep(inputStep);
    setPublishData(submissionData);
  }, [openPopout]); // eslint-disable-line react-hooks/exhaustive-deps

  React.useEffect(() => {
    if (open) {
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
  }, [open, api]);

  const handleClose = () => {
    setStep("select");
    setSelectedAgent(null);
    setPublishData({
      name: "",
      sub_heading: "",
      description: "",
      image_urls: [],
      agent_id: "",
      agent_version: 0,
      slug: "",
      categories: [],
    });
    setOpen(false);
  };

  const handleAgentSelect = (agentName: string) => {
    setSelectedAgent(agentName);
  };

  const handleNextFromSelect = (agentId: string, agentVersion: number) => {
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

    setStep("info");
    setSelectedAgentId(agentId);
    setSelectedAgentVersion(agentVersion);
  };

  const handleNextFromInfo = async (
    name: string,
    subHeading: string,
    slug: string,
    description: string,
    imageUrls: string[],
    videoUrl: string,
    categories: string[],
  ) => {
    const missingFields: string[] = [];

    if (!name) missingFields.push("Name");
    if (!subHeading) missingFields.push("Sub-heading");
    if (!description) missingFields.push("Description");
    if (!imageUrls.length) missingFields.push("Image");
    if (!categories.length) missingFields.push("Categories");

    if (missingFields.length > 0) {
      toast({
        title: "Missing Required Fields",
        description: `Please fill in: ${missingFields.join(", ")}`,
        duration: 3000,
      });
      return;
    }

    setPublishData({
      name,
      sub_heading: subHeading,
      description,
      image_urls: imageUrls,
      video_url: videoUrl,
      agent_id: selectedAgentId || "",
      agent_version: selectedAgentVersion || 0,
      slug,
      categories,
    });

    // Create store submission
    try {
      const submission = await api.createStoreSubmission({
        name: name,
        sub_heading: subHeading,
        description: description,
        image_urls: imageUrls,
        video_url: videoUrl,
        agent_id: selectedAgentId || "",
        agent_version: selectedAgentVersion || 0,
        slug: slug.replace(/\s+/g, "-"),
        categories: categories,
      });
    } catch (error) {
      console.error("Error creating store submission:", error);
    }
    setStep("review");
  };

  const handleBack = () => {
    if (step === "info") {
      setStep("select");
    } else if (step === "review") {
      setStep("info");
    }
  };

  const renderContent = () => {
    switch (step) {
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
        return publishData ? (
          <PublishAgentAwaitingReview
            agentName={publishData.name}
            subheader={publishData.sub_heading}
            description={publishData.description}
            thumbnailSrc={publishData.image_urls[0]}
            onClose={handleClose}
            onDone={handleClose}
            onViewProgress={() => {
              router.push("/profile/dashboard");
              handleClose();
            }}
          />
        ) : null;
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger || <Button>Publish Agent</Button>}
      </DialogTrigger>
      <DialogContent className="h-screen w-screen max-w-full overflow-auto rounded-none border-none bg-black/40 backdrop-blur-[0.375rem]">
        {renderContent()}
      </DialogContent>
    </Dialog>
  );
};

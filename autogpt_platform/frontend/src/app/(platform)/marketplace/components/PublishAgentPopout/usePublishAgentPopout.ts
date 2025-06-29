import { useEffect, useId, useState } from "react";
import {
  MyAgentsResponse,
  StoreSubmissionRequest,
} from "@/lib/autogpt-server-api";
import { PublishAgentInfoInitialData } from "../PublishAgentInfo/PublishAgentSelectInfo";
import { useRouter } from "next/navigation";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToast } from "@/components/ui/use-toast";

interface usePublishAgentPopout {
  openPopout: boolean;
  inputStep: "select" | "info" | "review";
  submissionData: StoreSubmissionRequest;
}

export const usePublishAgentPopout = ({
  openPopout,
  inputStep,
  submissionData,
}: usePublishAgentPopout) => {
  const { toast } = useToast();

  const [step, setStep] = useState<"select" | "info" | "review">(inputStep);
  const [myAgents, setMyAgents] = useState<MyAgentsResponse | null>(null);
  const [_, setSelectedAgent] = useState<string | null>(null);
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
  const [publishData, setPublishData] =
    useState<StoreSubmissionRequest>(submissionData);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [selectedAgentVersion, setSelectedAgentVersion] = useState<
    number | null
  >(null);
  const [open, setOpen] = useState(false);

  const popupId = useId();
  const router = useRouter();
  const api = useBackendAPI();

  useEffect(() => {
    setOpen(openPopout);
    setStep(inputStep);
    setPublishData(submissionData);
  }, [openPopout]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
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
    setPublishData({
      name,
      sub_heading: subHeading,
      description,
      image_urls: imageUrls,
      video_url: videoUrl,
      agent_id: selectedAgentId || "",
      agent_version: selectedAgentVersion || 0,
      slug,
      categories: filteredCategories,
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

  return {
    handleBack,
    handleNextFromInfo,
    handleNextFromSelect,
    handleAgentSelect,
    handleClose,
    router,
    step,
    myAgents,
    selectedAgent: _,
    initialData,
    publishData,
    open,
    setOpen,
    popupId,
  };
};

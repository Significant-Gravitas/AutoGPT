"use client";

import * as React from "react";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverAnchor,
} from "@/components/ui/popover";
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
  const [currentPage, setCurrentPage] = React.useState(1);
  const [totalPages, setTotalPages] = React.useState(1);
  const [loading, setLoading] = React.useState(false);

  const popupId = React.useId();
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
      setCurrentPage(1);
    }
  }, [open]);

  React.useEffect(() => {
    if (open) {
      const loadMyAgents = async () => {
        try {
          setLoading(true);
          const response = await api.getMyAgents({ page: currentPage, page_size: 20 });
          setMyAgents(response);
          setCurrentPage(response.pagination.current_page);
          setTotalPages(response.pagination.total_pages);
        } catch (error) {
          console.error("Failed to load my agents:", error);
        } finally {
          setLoading(false);
        }
      };

      loadMyAgents();
    }
  }, [open, currentPage, api]);

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
          <div className="flex min-h-screen items-center justify-center">
            <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg dark:bg-gray-800">
              <div className="h-full overflow-y-auto">
                {loading ? (
                  <div className="flex items-center justify-center p-8 text-gray-600">
                    Loading agents...
                  </div>
                ) : (
                  <>
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
                    <div className="flex justify-between items-center p-4">
                      <button
                        onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                        disabled={currentPage === 1}
                        className="text-sm font-medium text-blue-600 disabled:text-gray-400"
                      >
                        ← Previous
                      </button>
                      <span className="text-sm text-gray-700">
                        Page {currentPage} of {totalPages}
                      </span>
                      <button
                        onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                        disabled={currentPage === totalPages}
                        className="text-sm font-medium text-blue-600 disabled:text-gray-400"
                      >
                        Next →
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        );
      case "info":
        return (
          <div className="flex min-h-screen items-center justify-center">
            <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg dark:bg-gray-800">
              <div className="h-[700px] overflow-y-auto">
                <PublishAgentInfo
                  onBack={handleBack}
                  onSubmit={handleNextFromInfo}
                  onClose={handleClose}
                  initialData={initialData}
                />
              </div>
            </div>
          </div>
        );
      case "review":
        return publishData ? (
          <div className="flex justify-center">
            <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg dark:bg-gray-800">
              <div className="h-[600px] overflow-y-auto">
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
              </div>
            </div>
          </div>
        ) : null;
    }
  };

  return (
    <Popover
      open={open}
      onOpenChange={(isOpen) => {
        if (isOpen !== open) {
          setOpen(isOpen);
        }
      }}
    >
      <PopoverTrigger asChild>
        {trigger || <Button>Publish Agent</Button>}
      </PopoverTrigger>
      <PopoverAnchor asChild>
        <div className="fixed left-0 top-0 hidden h-screen w-screen items-center justify-center"></div>
      </PopoverAnchor>

      <PopoverContent
        id={popupId}
        align="center"
        className="z-50 h-screen w-screen bg-transparent"
      >
        {renderContent()}
      </PopoverContent>
    </Popover>
  );
};

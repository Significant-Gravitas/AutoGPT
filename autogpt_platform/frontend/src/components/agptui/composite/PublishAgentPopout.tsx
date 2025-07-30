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
import { MyAgent } from "@/app/api/__generated__/models/myAgent";
import { useRouter } from "next/navigation";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useToast } from "@/components/molecules/Toast/use-toast";
import LoadingBox, { LoadingSpinner } from "@/components/ui/loading";
import { StoreSubmissionRequest } from "@/app/api/__generated__/models/storeSubmissionRequest";
import { useGetV2GetMyAgents } from "@/app/api/__generated__/endpoints/store/store";
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
  const [allAgents, setAllAgents] = React.useState<MyAgent[]>([]);
  const [_, setSelectedAgent] = React.useState<string | null>(null);
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
  const [loadingMore, setLoadingMore] = React.useState(false);
  const [hasMore, setHasMore] = React.useState(true);

  const api = useBackendAPI();

  // Use the auto-generated API hook
  const { data, error, isLoading, refetch } = useGetV2GetMyAgents({
    request: {
      params: {
        page: currentPage,
        page_size: 20,
      },
    },
    query: {
      enabled: open, // Only fetch when the popout is open
    },
  });

  // Update allAgents when new data arrives
  React.useEffect(() => {
    if (data?.status === 200 && data.data) {
      if (currentPage === 1) {
        setAllAgents(data.data.agents);
      } else {
        setAllAgents((prev) => [...prev, ...data.data.agents]);
      }
      setHasMore(
        data.data.pagination.current_page < data.data.pagination.total_pages,
      );
    }
  }, [data, currentPage]);

  const fetchMyAgents = React.useCallback(
    async (page: number, append = false) => {
      if (append) {
        setLoadingMore(true);
        setCurrentPage(page);
      } else {
        setCurrentPage(page);
        setAllAgents([]);
      }
    },
    [],
  );

  const popupId = React.useId();
  const router = useRouter();

  const { toast } = useToast();

  React.useEffect(() => {
    setOpen(openPopout);
    setStep(inputStep);
    setPublishData(submissionData);
  }, [openPopout]); // eslint-disable-line react-hooks/exhaustive-deps

  React.useEffect(() => {
    if (open) {
      setCurrentPage(1);
      setHasMore(true);
      setAllAgents([]);
    }
  }, [open]);

  // Handle loading state for pagination
  React.useEffect(() => {
    if (currentPage > 1 && !isLoading) {
      setLoadingMore(false);
    }
  }, [currentPage, isLoading]);

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
    const selectedAgentData = allAgents.find(
      (agent) => agent.agent_id === agentId,
    ) as any;

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

  const handleScroll = React.useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
      if (
        hasMore &&
        !loadingMore &&
        scrollTop + clientHeight >= scrollHeight - 50
      ) {
        fetchMyAgents(currentPage + 1, true);
      }
    },
    [hasMore, loadingMore, currentPage],
  );

  const renderContent = () => {
    switch (step) {
      case "select":
        return (
          <div className="flex min-h-screen items-center justify-center">
            <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg dark:bg-gray-800">
              <div className="h-full overflow-y-hidden">
                {isLoading && currentPage === 1 ? (
                  <LoadingBox className="p-8" />
                ) : error ? (
                  <div className="flex flex-col items-center justify-center gap-4 p-8">
                    <p className="text-red-600">
                      Failed to load agents. Please try again.
                    </p>
                    <Button
                      onClick={() => {
                        refetch();
                      }}
                      variant="outline"
                    >
                      Try Again
                    </Button>
                  </div>
                ) : (
                  <>
                    <PublishAgentSelect
                      agents={
                        allAgents
                          .map((agent) => ({
                            name: agent.agent_name,
                            id: agent.agent_id,
                            version: agent.agent_version,
                            lastEdited: agent.last_edited,
                            imageSrc:
                              agent.agent_image ||
                              "https://picsum.photos/300/200",
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
                      onListScroll={handleScroll}
                    />
                    {loadingMore && (
                      <div className="flex items-center justify-center p-4">
                        <LoadingSpinner className="size-6" />
                      </div>
                    )}
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
                  description={publishData.description || ""}
                  thumbnailSrc={publishData.image_urls?.[0]}
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

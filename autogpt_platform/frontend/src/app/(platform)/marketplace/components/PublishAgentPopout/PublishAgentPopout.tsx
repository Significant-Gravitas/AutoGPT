"use client";

import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverAnchor,
} from "@/components/ui/popover";
import { PublishAgentSelect } from "../PublishAgentSelect/PublishAgentSelect";
import { PublishAgentInfo } from "../PublishAgentInfo/PublishAgentSelectInfo";
import { PublishAgentAwaitingReview } from "../PublishAgentAwaitingReview/PublishAgentAwaitingReview";
import { Button } from "../../../../../components/agptui/Button";
import { StoreSubmissionRequest } from "@/lib/autogpt-server-api";
import { usePublishAgentPopout } from "./usePublishAgentPopout";

interface PublishAgentPopoutProps {
  trigger?: React.ReactNode;
  openPopout?: boolean;
  inputStep?: "select" | "info" | "review";
  submissionData?: StoreSubmissionRequest;
}

export const PublishAgentPopout = ({
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
}: PublishAgentPopoutProps) => {
  const {
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
  } = usePublishAgentPopout({ openPopout, inputStep, submissionData });

  const renderContent = () => {
    switch (step) {
      case "select":
        return (
          <div className="flex min-h-screen items-center justify-center">
            <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg dark:bg-gray-800">
              <div className="h-full overflow-y-auto">
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

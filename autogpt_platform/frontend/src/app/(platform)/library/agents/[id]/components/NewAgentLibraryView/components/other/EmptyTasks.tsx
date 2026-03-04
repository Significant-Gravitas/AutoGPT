"use client";

import { getV1GetGraphVersion } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2ListLibraryAgentsQueryKey,
  useDeleteV2DeleteLibraryAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ShowMoreText } from "@/components/molecules/ShowMoreText/ShowMoreText";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { exportAsJSONFile } from "@/lib/utils";
import { formatDate } from "@/lib/utils/time";
import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { RunAgentModal } from "../modals/RunAgentModal/RunAgentModal";
import { RunDetailCard } from "../selected-views/RunDetailCard/RunDetailCard";
import { EmptyTasksIllustration } from "./EmptyTasksIllustration";

type Props = {
  agent: LibraryAgent;
  onRun?: (run: GraphExecutionMeta) => void;
  onTriggerSetup?: (preset: LibraryAgentPreset) => void;
  onScheduleCreated?: (schedule: GraphExecutionJobInfo) => void;
};

export function EmptyTasks({
  agent,
  onRun,
  onTriggerSetup,
  onScheduleCreated,
}: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const router = useRouter();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [isDeletingAgent, setIsDeletingAgent] = useState(false);

  const { mutateAsync: deleteAgent } = useDeleteV2DeleteLibraryAgent();

  async function handleDeleteAgent() {
    if (!agent.id) return;

    setIsDeletingAgent(true);

    try {
      await deleteAgent({ libraryAgentId: agent.id });

      await queryClient.refetchQueries({
        queryKey: getGetV2ListLibraryAgentsQueryKey(),
      });

      toast({ title: "Agent deleted" });
      setShowDeleteDialog(false);
      router.push("/library");
    } catch (error: unknown) {
      toast({
        title: "Failed to delete agent",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsDeletingAgent(false);
    }
  }

  async function handleExport() {
    try {
      const res = await getV1GetGraphVersion(
        agent.graph_id,
        agent.graph_version,
        { for_export: true },
      );
      if (res.status === 200) {
        const filename = `${agent.name}_v${agent.graph_version}.json`;
        exportAsJSONFile(res.data as any, filename);
        toast({ title: "Agent exported" });
      } else {
        toast({ title: "Failed to export agent", variant: "destructive" });
      }
    } catch (e: any) {
      toast({
        title: "Failed to export agent",
        description: e?.message,
        variant: "destructive",
      });
    }
  }
  const isPublished = Boolean(agent.marketplace_listing);
  const createdAt = formatDate(agent.created_at);
  const updatedAt = formatDate(agent.updated_at);
  const isUpdated = updatedAt !== createdAt;

  return (
    <div className="my-4 flex min-h-0 flex-1 flex-col gap-2 px-2 lg:flex-row">
      <RunDetailCard className="relative flex min-h-0 flex-1 flex-col overflow-hidden border-none">
        <div className="flex flex-1 flex-col items-center justify-center gap-0">
          <EmptyTasksIllustration className="-mt-20" />
          <div className="flex flex-col items-center gap-12">
            <div className="flex items-center justify-between gap-2">
              <div className="flex flex-col items-center gap-2">
                <Text variant="h3" className="text-center text-[1.375rem]">
                  Ready to get started?
                </Text>
                <Text variant="large" className="text-center">
                  Run your agent and this space will fill with your agent&apos;s
                  activity
                </Text>
              </div>
            </div>
            <RunAgentModal
              triggerSlot={
                <Button
                  variant="primary"
                  size="large"
                  className="inline-flex w-[19.75rem]"
                >
                  Setup your task
                </Button>
              }
              agent={agent}
              onRunCreated={onRun}
              onTriggerSetup={onTriggerSetup}
              onScheduleCreated={onScheduleCreated}
            />
          </div>
        </div>
      </RunDetailCard>

      <div className="mt-4 flex flex-col gap-10 rounded-large border border-zinc-200 p-6 lg:mt-0 lg:w-[29.5rem]">
        <Text variant="label" className="text-zinc-500">
          About this agent
        </Text>
        <div className="flex flex-col gap-2">
          <Text variant="h4">{agent.name}</Text>
          {isPublished ? (
            <Text variant="body">
              by {agent.marketplace_listing?.creator.name}
            </Text>
          ) : null}
        </div>
        <ShowMoreText
          previewLimit={170}
          variant="body"
          className="-mt-4 text-textGrey"
        >
          {agent.description ||
            `This agent is not yet published. Once it is published, You can publish your agent by clicking the "Publish" button in the agent editor.`}
        </ShowMoreText>
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-20">
            <div className="flex flex-col gap-0">
              <Text variant="body-medium" className="text-black">
                Agent created on
              </Text>
              <Text variant="body" className="text-textGrey">
                {createdAt}
              </Text>
            </div>
            {isUpdated ? (
              <div className="flex flex-col gap-0">
                <Text variant="body-medium" className="text-black">
                  Agent updated on
                </Text>
                <Text variant="body" className="text-textGrey">
                  {updatedAt}
                </Text>
              </div>
            ) : null}
          </div>
          <div className="mt-4 flex items-center gap-2">
            <Button variant="secondary" size="small" asChild>
              <Link
                href={`/build?flowID=${agent.graph_id}&flowVersion=${agent.graph_version}`}
                target="_blank"
              >
                Edit agent
              </Link>
            </Button>
            <Button variant="secondary" size="small" onClick={handleExport}>
              Export agent to file
            </Button>
            <Button
              variant="secondary"
              size="small"
              onClick={() => setShowDeleteDialog(true)}
            >
              Delete agent
            </Button>
          </div>
        </div>
      </div>

      <Dialog
        controlled={{
          isOpen: showDeleteDialog,
          set: setShowDeleteDialog,
        }}
        styling={{ maxWidth: "32rem" }}
        title="Delete agent"
      >
        <Dialog.Content>
          <div>
            <Text variant="large">
              Are you sure you want to delete this agent? This action cannot be
              undone.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeletingAgent}
                onClick={() => setShowDeleteDialog(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDeleteAgent}
                loading={isDeletingAgent}
              >
                Delete Agent
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}

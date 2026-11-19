"use client";

import { RunAgentModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentModal/RunAgentModal";
import {
  getGetV1ListGraphExecutionsQueryKey,
  usePostV1ExecuteGraphAgent,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { useQueryClient } from "@tanstack/react-query";
import { needsSetup } from "../runs";

interface Props {
  agent: LibraryAgent | undefined;
  onStarted: (executionID: string) => void;
  /** Opens the run form in the tab, for a workflow that asks for inputs or
   *  an account. */
  onSetup: () => void;
}

/** Run the workflow now. One that asks for inputs or an account opens the
 *  run form in the tab first; any other starts at once and its run shows
 *  live. A workflow started by a trigger keeps the trigger dialog. */
export function RunNowButton({ agent, onStarted, onSetup }: Props) {
  const queryClient = useQueryClient();
  const execute = usePostV1ExecuteGraphAgent({
    mutation: {
      onSuccess: (res) => {
        if (res.status !== 200) return;
        queryClient.invalidateQueries({
          queryKey: getGetV1ListGraphExecutionsQueryKey(res.data.graph_id),
        });
        onStarted(res.data.id);
      },
      onError: (error) =>
        toast({
          title: "The workflow could not start",
          description: error instanceof Error ? error.message : undefined,
          variant: "destructive",
        }),
    },
  });

  const button = (
    <Button
      variant="primary"
      size="xs"
      leadingIcon={PlayIcon}
      disabled={!agent}
      loading={execute.isPending}
      onClick={
        agent && !needsSetup(agent)
          ? () =>
              execute.mutate({
                graphId: agent.graph_id,
                graphVersion: agent.graph_version,
                data: { inputs: {}, credentials_inputs: {}, source: "library" },
              })
          : agent && !agent.trigger_setup_info
            ? onSetup
            : undefined
      }
    >
      Run now
    </Button>
  );

  if (!agent?.trigger_setup_info) return button;
  return (
    <RunAgentModal
      agent={agent}
      title="Run workflow"
      dialogVariant="compact"
      triggerSlot={button}
      onRunCreated={(execution) => onStarted(execution.id)}
    />
  );
}

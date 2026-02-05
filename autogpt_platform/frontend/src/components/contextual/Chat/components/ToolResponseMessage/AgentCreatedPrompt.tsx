"use client";

import { useGetV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { RunAgentModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentModal/RunAgentModal";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  CheckCircleIcon,
  PencilLineIcon,
  PlayIcon,
} from "@phosphor-icons/react";
import { AIChatBubble } from "../AIChatBubble/AIChatBubble";

interface Props {
  agentName: string;
  libraryAgentId: string;
  onSendMessage?: (content: string) => void;
}

export function AgentCreatedPrompt({
  agentName,
  libraryAgentId,
  onSendMessage,
}: Props) {
  // Fetch library agent eagerly so modal is ready when user clicks
  const { data: libraryAgentResponse, isLoading } = useGetV2GetLibraryAgent(
    libraryAgentId,
    {
      query: {
        enabled: !!libraryAgentId,
      },
    },
  );

  const libraryAgent =
    libraryAgentResponse?.status === 200 ? libraryAgentResponse.data : null;

  function handleRunWithPlaceholders() {
    onSendMessage?.(
      `Run the agent "${agentName}" with placeholder/example values so I can test it.`,
    );
  }

  function handleRunCreated(execution: GraphExecutionMeta) {
    onSendMessage?.(
      `I've started the agent "${agentName}". The execution ID is ${execution.id}. Please monitor its progress and let me know when it completes.`,
    );
  }

  function handleScheduleCreated(schedule: GraphExecutionJobInfo) {
    const scheduleInfo = schedule.cron
      ? `with cron schedule "${schedule.cron}"`
      : "to run on the specified schedule";
    onSendMessage?.(
      `I've scheduled the agent "${agentName}" ${scheduleInfo}. The schedule ID is ${schedule.id}.`,
    );
  }

  return (
    <AIChatBubble>
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-green-100">
            <CheckCircleIcon
              size={18}
              weight="fill"
              className="text-green-600"
            />
          </div>
          <div>
            <Text variant="body-medium" className="text-neutral-900">
              Agent Created Successfully
            </Text>
            <Text variant="small" className="text-neutral-500">
              &quot;{agentName}&quot; is ready to test
            </Text>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <Text variant="small-medium" className="text-neutral-700">
            Ready to test?
          </Text>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="small"
              onClick={handleRunWithPlaceholders}
              className="gap-2"
            >
              <PlayIcon size={16} />
              Run with example values
            </Button>
            {libraryAgent ? (
              <RunAgentModal
                triggerSlot={
                  <Button variant="outline" size="small" className="gap-2">
                    <PencilLineIcon size={16} />
                    Run with my inputs
                  </Button>
                }
                agent={libraryAgent}
                onRunCreated={handleRunCreated}
                onScheduleCreated={handleScheduleCreated}
              />
            ) : (
              <Button
                variant="outline"
                size="small"
                loading={isLoading}
                disabled
                className="gap-2"
              >
                <PencilLineIcon size={16} />
                Run with my inputs
              </Button>
            )}
          </div>
        </div>
      </div>
    </AIChatBubble>
  );
}

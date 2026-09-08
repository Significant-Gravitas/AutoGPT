"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { cn } from "@/lib/utils";
import { Activity01Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import { ExpertWorkflowActions } from "./ExpertWorkflowActions";
import { ExpertWorkflowRunButton } from "./ExpertWorkflowRunButton";
import { useExpertWorkflowCard } from "./useExpertWorkflowCard";
import { WorkflowChain } from "./WorkflowChain";
import { WorkflowCredentialStack } from "./WorkflowCredentialStack";

interface Props {
  workflow: ExpertWorkflowRef;
  expertId?: string;
  coverColor: string | undefined;
  onAsk?: (prompt: string) => void;
}

export function ExpertWorkflowCard({
  workflow,
  expertId,
  coverColor,
  onAsk,
}: Props) {
  const {
    name,
    libraryAgent,
    runCount,
    isTriggerWorkflow,
    status,
    credentialProviders,
    libraryHref,
    builderHref,
    chatHref,
    chatPrompt,
    openRun,
    openTriggers,
  } = useExpertWorkflowCard({ workflow, expertId });

  return (
    <div
      data-testid="expert-workflow-row"
      className="group relative flex w-full flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white transition-colors hover:border-zinc-300"
    >
      {libraryHref ? (
        <NextLink
          href={libraryHref}
          aria-label={`Open ${name} tasks`}
          className="absolute inset-0 z-0 rounded-xl"
        />
      ) : null}

      <div className="pointer-events-none relative mx-1.5 mt-1.5 flex h-32 items-center justify-center overflow-hidden rounded-lg bg-zinc-100">
        <ExpertCover
          className="absolute inset-0 h-full w-full rounded-none"
          color={coverColor}
        />
        <div className="relative">
          <WorkflowChain chain={workflow.chain ?? []} size="sm" />
        </div>
        <div className="pointer-events-auto absolute right-2 top-2 z-10 flex items-center gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 has-[[data-state=open]]:opacity-100">
          <ExpertWorkflowActions
            workflow={workflow}
            expertId={expertId}
            name={name}
            builderHref={builderHref}
            chatHref={chatHref}
            chatPrompt={chatPrompt}
            onAsk={onAsk}
          />
        </div>
      </div>

      <div className="pointer-events-none relative flex flex-1 flex-col px-3 pb-3 pt-2.5">
        <Text
          variant="large-medium"
          tone="primary"
          className="line-clamp-2 hyphens-auto break-words"
        >
          {name}
        </Text>
        {workflow.schedule_cron ? (
          <Text variant="small" tone="muted" className="mt-0.5">
            {safeHumanizeCronExpression(workflow.schedule_cron)}
          </Text>
        ) : null}
        {workflow.description ? (
          <Text
            variant="body"
            tone="muted"
            className="mt-1.5 line-clamp-2 leading-5"
          >
            {workflow.description}
          </Text>
        ) : null}
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {runCount !== undefined ? (
            <Text
              variant="body"
              tone="muted"
              className="flex items-center gap-1.5 leading-5"
            >
              <Icon icon={Activity01Icon} size={14} className="shrink-0" />
              {runCount} {runCount === 1 ? "run" : "runs"}
            </Text>
          ) : null}
          <Text
            variant="small-medium"
            as="span"
            className={cn(
              "rounded-md px-2 py-0.5 ring-1 ring-inset ring-zinc-200/80",
              status.className,
            )}
          >
            {status.label}
          </Text>
        </div>
        {libraryAgent ? (
          <div className="relative z-10 mt-auto flex items-center justify-between gap-3 pt-3">
            <WorkflowCredentialStack providers={credentialProviders} />
            <span className="pointer-events-auto ml-auto">
              <ExpertWorkflowRunButton
                agent={libraryAgent}
                isTriggerWorkflow={isTriggerWorkflow}
                onRunCreated={openRun}
                onTriggerSetup={openTriggers}
              />
            </span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

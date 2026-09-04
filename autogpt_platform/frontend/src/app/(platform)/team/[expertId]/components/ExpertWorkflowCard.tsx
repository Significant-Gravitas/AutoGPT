"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { cn } from "@/lib/utils";
import { Activity01Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { ExpertCover } from "../../components/ExpertTeamCard/components/ExpertCover";
import {
  ExpertWorkflowActions,
  ExpertWorkflowRunButton,
} from "./ExpertWorkflowActions";
import { useExpertWorkflowCard } from "./useExpertWorkflowCard";
import { WorkflowChain } from "./WorkflowChain";
import { WorkflowCredentialStack } from "./WorkflowCredentialStack";

interface Props {
  workflow: ExpertWorkflowRef;
  expertId: string;
  coverColor: string | undefined;
}

export function ExpertWorkflowCard({ workflow, expertId, coverColor }: Props) {
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
    openRun,
    openTriggers,
  } = useExpertWorkflowCard({ workflow, expertId });

  return (
    <div
      data-testid="expert-workflow-row"
      className="group relative flex w-full flex-col overflow-hidden rounded-[1.75rem] border border-zinc-100 bg-white transition-shadow hover:shadow-md"
    >
      {libraryHref ? (
        <NextLink
          href={libraryHref}
          aria-label={`Open ${name} tasks`}
          className="absolute inset-0 z-0 rounded-[1.75rem]"
        />
      ) : null}

      <div className="pointer-events-none relative mx-1.5 mt-1.5 flex h-36 items-center justify-center overflow-hidden rounded-[1.375rem] bg-zinc-100">
        <ExpertCover
          className="absolute inset-0 h-full w-full rounded-none"
          color={coverColor}
        />
        <div className="relative">
          <WorkflowChain chain={workflow.chain ?? []} />
        </div>
        <div className="pointer-events-auto absolute right-2 top-2 z-10 flex items-center gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 has-[[data-state=open]]:opacity-100">
          <ExpertWorkflowActions
            workflow={workflow}
            expertId={expertId}
            name={name}
            builderHref={builderHref}
            chatHref={chatHref}
          />
        </div>
      </div>

      <div className="pointer-events-none relative flex flex-1 flex-col px-3 pb-3 pt-2.5">
        <Text variant="h5" className="line-clamp-2 hyphens-auto break-words">
          {name}
        </Text>
        {workflow.schedule_cron ? (
          <Text variant="small" className="mt-0.5 text-zinc-400">
            {safeHumanizeCronExpression(workflow.schedule_cron)}
          </Text>
        ) : null}
        {workflow.description ? (
          <Text
            variant="small"
            className="mt-1.5 line-clamp-2 text-sm leading-5 text-zinc-500"
          >
            {workflow.description}
          </Text>
        ) : null}
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {runCount !== undefined ? (
            <Text
              variant="small"
              className="flex items-center gap-1.5 text-sm leading-5 text-zinc-500"
            >
              <Icon icon={Activity01Icon} size={14} className="shrink-0" />
              {runCount} {runCount === 1 ? "run" : "runs"}
            </Text>
          ) : null}
          <span
            className={cn(
              "rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ring-zinc-200/80",
              status.className,
            )}
          >
            {status.label}
          </span>
        </div>
        {libraryAgent ? (
          <div className="pointer-events-auto relative z-10 mt-auto flex items-center justify-between gap-3 pt-3">
            <WorkflowCredentialStack providers={credentialProviders} />
            <span className="ml-auto">
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

"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { cn } from "@/lib/utils";
import { Activity01Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import {
  ExpertWorkflowActions,
  ExpertWorkflowRunButton,
} from "./ExpertWorkflowActions";
import { useExpertWorkflowCard } from "./useExpertWorkflowCard";
import { WorkflowCredentialStack } from "./WorkflowCredentialStack";

interface Props {
  workflow: ExpertWorkflowRef;
  expertId: string;
  accentClassName: string;
}

export function ExpertWorkflowListItem({
  workflow,
  expertId,
  accentClassName,
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
    openRun,
    openTriggers,
  } = useExpertWorkflowCard({ workflow, expertId });
  const meta = [
    workflow.schedule_cron
      ? safeHumanizeCronExpression(workflow.schedule_cron)
      : null,
    runCount !== undefined
      ? `${runCount} ${runCount === 1 ? "run" : "runs"}`
      : null,
  ].filter((part): part is string => Boolean(part));

  return (
    <div
      data-testid="expert-workflow-row"
      className="group relative flex items-center gap-4 rounded-lg border border-zinc-200 bg-white px-3 py-2.5 transition-colors hover:bg-zinc-50"
    >
      {libraryHref ? (
        <NextLink
          href={libraryHref}
          aria-label={`Open ${name} tasks`}
          className="absolute inset-0 z-0 rounded-lg"
        />
      ) : null}

      <div
        className={cn(
          accentClassName,
          "pointer-events-none flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-md border-0",
        )}
      >
        <Icon
          icon={status.icon}
          size={18}
          role="img"
          aria-label={status.label}
        />
      </div>

      <div className="pointer-events-none min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <Text variant="body-medium" className="truncate text-zinc-900">
            {name}
          </Text>
          <span
            className={cn(
              "shrink-0 rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset ring-zinc-200/80",
              status.className,
            )}
          >
            {status.label}
          </span>
        </div>
        {workflow.description ? (
          <Text
            variant="small"
            className="mt-0.5 truncate text-sm text-zinc-500"
          >
            {workflow.description}
          </Text>
        ) : null}
        {meta.length > 0 || credentialProviders.length > 0 ? (
          <div className="mt-1 flex flex-wrap items-center gap-2 text-sm leading-5 text-zinc-600">
            {meta.length > 0 ? (
              <Text
                variant="small"
                className="flex items-center gap-1.5 text-sm leading-5 text-zinc-600"
              >
                <Icon icon={Activity01Icon} size={14} className="shrink-0" />
                {meta.join(" · ")}
              </Text>
            ) : null}
            {meta.length > 0 && credentialProviders.length > 0 ? (
              <span className="text-zinc-300">•</span>
            ) : null}
            <WorkflowCredentialStack providers={credentialProviders} />
          </div>
        ) : null}
      </div>

      <div className="pointer-events-auto relative z-10 flex shrink-0 items-center gap-1">
        <ExpertWorkflowActions
          workflow={workflow}
          expertId={expertId}
          name={name}
          builderHref={builderHref}
          chatHref={chatHref}
          buttonClassName="h-8 w-8 rounded-md border-transparent p-0 text-zinc-700 hover:border-transparent hover:bg-zinc-50"
        />
        {libraryAgent ? (
          <span className="ml-1">
            <ExpertWorkflowRunButton
              agent={libraryAgent}
              isTriggerWorkflow={isTriggerWorkflow}
              variant="secondary"
              onRunCreated={openRun}
              onTriggerSetup={openTriggers}
            />
          </span>
        ) : null}
      </div>
    </div>
  );
}

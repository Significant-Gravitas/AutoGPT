"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { cn } from "@/lib/utils";
import { Activity01Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { ExpertWorkflowActions } from "./ExpertWorkflowActions";
import { ExpertWorkflowRunButton } from "./ExpertWorkflowRunButton";
import { useExpertWorkflowCard } from "./useExpertWorkflowCard";
import { WorkflowCredentialStack } from "./WorkflowCredentialStack";

interface Props {
  workflow: ExpertWorkflowRef;
  libraryAgent: LibraryAgent | undefined;
  expertId?: string;
  onAsk?: (prompt: string) => void;
}

export function ExpertWorkflowListItem({
  workflow,
  libraryAgent,
  expertId,
  onAsk,
}: Props) {
  const {
    name,
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
  } = useExpertWorkflowCard({
    workflow,
    libraryAgent,
    expertId,
  });
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
      className="group relative flex flex-col items-stretch gap-3 rounded-2xl bg-surface px-3.5 py-2.5 transition-colors smooth-shadow-ring-sm hover:bg-surface-muted sm:flex-row sm:items-center sm:gap-4"
    >
      {libraryHref ? (
        <NextLink
          href={libraryHref}
          aria-label={`Open ${name} tasks`}
          className="absolute inset-0 z-0 rounded-lg"
        />
      ) : null}

      <div className="pointer-events-none min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <Text variant="body-md-medium" tone="primary" className="truncate">
            {name}
          </Text>
          <Text
            variant="small-medium"
            as="span"
            className={cn(
              "shrink-0 rounded-md px-2 py-0.5 ring-1 ring-inset ring-border-base/80",
              status.className,
            )}
          >
            {status.label}
          </Text>
        </div>
        {workflow.description ? (
          <Text variant="body-md" tone="muted" className="mt-0.5 truncate">
            {workflow.description}
          </Text>
        ) : null}
        {meta.length > 0 || credentialProviders.length > 0 ? (
          <div className="mt-1 flex flex-wrap items-center gap-2">
            {meta.length > 0 ? (
              <Text
                variant="body-md"
                tone="secondary"
                className="flex items-center gap-1.5"
              >
                <Icon icon={Activity01Icon} size={14} className="shrink-0" />
                {meta.join(" · ")}
              </Text>
            ) : null}
            {meta.length > 0 && credentialProviders.length > 0 ? (
              <Text variant="body-md" as="span" tone="disabled">
                •
              </Text>
            ) : null}
            <WorkflowCredentialStack providers={credentialProviders} />
          </div>
        ) : null}
      </div>

      <div className="pointer-events-auto relative z-10 flex shrink-0 items-center justify-end gap-1">
        <ExpertWorkflowActions
          workflow={workflow}
          expertId={expertId}
          name={name}
          builderHref={builderHref}
          chatHref={chatHref}
          chatPrompt={chatPrompt}
          onAsk={onAsk}
          variant="ghost"
          size="icon-sm"
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

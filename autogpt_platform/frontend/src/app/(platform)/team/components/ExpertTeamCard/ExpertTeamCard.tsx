import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { PlusIcon } from "@phosphor-icons/react";
import { MouseEvent } from "react";

import {
  getLastRunLabel,
  getScheduleSummary,
  getSpendLabel,
  workflowNeedsSetup,
} from "./helpers";
import { useExpertTeamCard } from "./useExpertTeamCard";

interface Props {
  expert: Expert;
  onInstallWorkflow: (expertId: string) => void;
  onOpenProfile: (expertId: string) => void;
}

export function ExpertTeamCard({
  expert,
  onInstallWorkflow,
  onOpenProfile,
}: Props) {
  const workflowCount = expert.workflows.length;
  const { handleResume, isResuming } = useExpertTeamCard(expert.id);
  const statusLine = [
    getScheduleSummary(expert),
    getLastRunLabel(expert),
    getSpendLabel(expert),
  ]
    .filter(Boolean)
    .join(" · ");
  const isPaused = Boolean(expert.schedules_paused_at);

  function handleInstallClick(event: MouseEvent) {
    event.stopPropagation();
    onInstallWorkflow(expert.id);
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onOpenProfile(expert.id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpenProfile(expert.id);
        }
      }}
      className="flex cursor-pointer flex-col gap-3 rounded-2xl border border-zinc-200 bg-white p-5 transition-all duration-200 hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)]"
    >
      <div className="flex items-center gap-3">
        <Avatar className="h-12 w-12">
          {expert.avatar_url ? (
            <AvatarImage src={expert.avatar_url} alt={expert.name} />
          ) : null}
          <AvatarFallback>{expert.name}</AvatarFallback>
        </Avatar>
        <div className="min-w-0 flex-1">
          <Text variant="large-medium">{expert.name}</Text>
          <Text variant="small" className="text-zinc-500">
            {expert.role}
          </Text>
        </div>
      </div>
      {statusLine ? (
        <Text variant="small" className="text-zinc-500">
          {statusLine}
        </Text>
      ) : null}
      {isPaused ? (
        <div className="flex items-center justify-between gap-2 rounded-xl bg-amber-50 px-3 py-2 ring-1 ring-inset ring-amber-200">
          <Text variant="small" className="text-amber-700">
            Schedules paused
          </Text>
          <Button
            variant="secondary"
            size="small"
            loading={isResuming}
            onClick={(event) => {
              event.stopPropagation();
              handleResume();
            }}
          >
            Resume schedules
          </Button>
        </div>
      ) : null}
      {expert.skills && expert.skills.length > 0 ? (
        <div>
          <div className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.14em] text-zinc-400">
            Skills
          </div>
          <div className="flex flex-wrap gap-1.5">
            {expert.skills.slice(0, 3).map((skill) => (
              <span
                key={skill}
                className="rounded-full bg-zinc-50 px-2.5 py-1 text-xs font-medium text-zinc-500 ring-1 ring-inset ring-zinc-200/80"
              >
                {skill}
              </span>
            ))}
            {expert.skills.length > 3 ? (
              <span className="px-1 py-1 text-xs font-medium text-zinc-400">
                +{expert.skills.length - 3}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}
      <div className="flex flex-col gap-2">
        <Text variant="small" className="text-zinc-500">
          {workflowCount} {workflowCount === 1 ? "workflow" : "workflows"}
        </Text>
        {workflowCount > 0 ? (
          <div className="flex flex-wrap gap-1">
            {expert.workflows.map((workflow) =>
              workflow.name ? (
                <span
                  key={workflow.id}
                  className={
                    workflowNeedsSetup(workflow)
                      ? "rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 ring-1 ring-inset ring-amber-200"
                      : "rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-700"
                  }
                >
                  {workflow.name}
                  {workflowNeedsSetup(workflow) ? " · Needs setup" : null}
                </span>
              ) : null,
            )}
          </div>
        ) : null}
      </div>
      <div className="mt-auto flex gap-2">
        <Button
          as="NextLink"
          href={`/copilot?expertId=${expert.id}`}
          variant="secondary"
          size="small"
          onClick={(event) => event.stopPropagation()}
        >
          Chat
        </Button>
        <Button
          variant="ghost"
          size="small"
          leftIcon={<PlusIcon size={16} />}
          onClick={handleInstallClick}
        >
          Install workflow
        </Button>
      </div>
    </div>
  );
}

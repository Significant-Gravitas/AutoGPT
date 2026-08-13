import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  Logout03Icon,
  MoreVerticalIcon,
  PencilIcon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { MouseEvent } from "react";

import { Progress } from "@/components/atoms/Progress/Progress";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  getNeedsSetupCount,
  getScheduleCountLabel,
  getWeeklySpend,
} from "../../helpers";
import { FireExpertDialog } from "../FireExpertDialog/FireExpertDialog";
import { useExpertTeamCard } from "./useExpertTeamCard";

interface Props {
  expert: Expert;
  schedules: GraphExecutionJobInfo[];
  onInstallWorkflow: (expertId: string) => void;
  onEditSoul: (expertId: string) => void;
}

export function ExpertTeamCard({
  expert,
  schedules,
  onInstallWorkflow,
  onEditSoul,
}: Props) {
  const workflowCount = expert.workflows.length;
  const needsSetupCount = getNeedsSetupCount(expert);
  const scheduleLabel = getScheduleCountLabel(schedules);
  const weeklySpend = getWeeklySpend(expert);
  const { handleResume, isResuming, isFireOpen, openFire, closeFire } =
    useExpertTeamCard(expert.id);
  const isPaused = Boolean(expert.schedules_paused_at);

  function handleInstallClick() {
    onInstallWorkflow(expert.id);
  }

  function handleEditSoulClick(event: MouseEvent) {
    event.stopPropagation();
    onEditSoul(expert.id);
  }

  return (
    <div className="relative flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-white p-5 transition-all duration-200 hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)]">
      <div className="absolute right-3 top-3 z-10">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label={`${expert.name} actions`}
              data-testid="expert-card-actions"
              onClick={(event) => event.stopPropagation()}
              className="inline-flex h-8 w-8 items-center justify-center rounded-full text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
            >
              <Icon icon={MoreVerticalIcon} size={18} />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" sideOffset={6}>
            <DropdownMenuItem
              onSelect={openFire}
              className="flex cursor-pointer items-center gap-2 text-red-600 focus:bg-red-50 focus:text-red-700"
            >
              <Icon icon={Logout03Icon} size={14} />
              Fire {expert.name}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <Link
        href={`/team/${expert.id}`}
        aria-label={`View ${expert.name}`}
        className="flex flex-col gap-3 pr-8"
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
        <Text variant="body" className="line-clamp-2 min-h-12">
          {expert.tagline ?? ""}
        </Text>
        <div className="flex flex-col gap-1">
          <div className="flex items-baseline justify-between gap-2">
            <Text variant="small" className="text-zinc-500">
              Credits this week
            </Text>
            <Text variant="small" className="tabular-nums text-zinc-500">
              {weeklySpend
                ? `${weeklySpend.spent} / ${weeklySpend.budget}`
                : "No budget"}
            </Text>
          </div>
          <Progress
            value={weeklySpend?.spent ?? 0}
            max={weeklySpend?.budget ?? 1}
            className={cn("h-1.5", !weeklySpend && "opacity-50")}
          />
        </div>
        <Text variant="small" className="min-h-5 text-zinc-500">
          {scheduleLabel ?? "No schedules yet"}
        </Text>
        <div className="flex min-h-5 items-center gap-2">
          <Text variant="small" className="text-zinc-500">
            {workflowCount} {workflowCount === 1 ? "workflow" : "workflows"}
          </Text>
          {needsSetupCount > 0 ? (
            <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 ring-1 ring-inset ring-amber-200">
              {needsSetupCount} {needsSetupCount === 1 ? "needs" : "need"} setup
            </span>
          ) : null}
        </div>
      </Link>
      {isPaused ? (
        <div className="flex items-center justify-between gap-2 rounded-xl bg-amber-50 px-3 py-2 ring-1 ring-inset ring-amber-200">
          <Text variant="small" className="text-amber-700">
            Schedules paused
          </Text>
          <Button
            variant="secondary"
            size="small"
            loading={isResuming}
            onClick={handleResume}
          >
            Resume schedules
          </Button>
        </div>
      ) : null}
      <div className="mt-auto flex flex-wrap gap-2">
        <Button
          as="NextLink"
          href={`/copilot?expertId=${expert.id}`}
          variant="secondary"
          size="small"
        >
          Chat
        </Button>
        <Button
          variant="secondary"
          size="small"
          leftIcon={<Icon icon={PencilIcon} size={16} />}
          onClick={handleEditSoulClick}
        >
          Edit Soul
        </Button>
        <Button
          variant="ghost"
          size="small"
          leftIcon={<Icon icon={PlusSignIcon} size={16} />}
          onClick={handleInstallClick}
        >
          Install workflow
        </Button>
      </div>
      <FireExpertDialog
        expertId={expert.id}
        expertName={expert.name}
        open={isFireOpen}
        onClose={closeFire}
      />
    </div>
  );
}

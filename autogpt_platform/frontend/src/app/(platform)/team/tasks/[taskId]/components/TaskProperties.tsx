"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { TaskRunRef } from "@/app/api/__generated__/models/taskRunRef";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import {
  Clock01Icon,
  DollarCircleIcon,
  Flag01Icon,
  PlayCircleIcon,
  Progress02Icon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { OriginBadge } from "../../../components/OriginBadge/OriginBadge";
import type { IconSvgElement } from "@hugeicons/react";
import Link from "next/link";
import { ACTION_BUTTON_CLASS } from "../../../helpers";
import {
  formatElapsed,
  formatSpend,
  getTaskOrbVariant,
  isOpenTask,
} from "../../../task-helpers";
import { TaskStatusChip } from "../../../components/TaskStatusChip/TaskStatusChip";

interface Props {
  task: DelegatedTask;
  onCancel: () => void;
  isCancelling: boolean;
}

export function TaskProperties({ task, onCancel, isCancelling }: Props) {
  const runs = task.runs ?? [];

  return (
    <aside className="flex flex-col gap-3 lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:self-start lg:overflow-y-auto">
      <Card title="Properties">
        <Row label="Status" icon={Progress02Icon}>
          <span className="inline-flex items-center gap-1.5">
            <TaskStatusChip
              status={task.status}
              orbVariant={getTaskOrbVariant(task.id)}
            />
            {task.stale_at && isOpenTask(task) ? (
              <Badge variant="warning" size="small">
                Stale
              </Badge>
            ) : null}
          </span>
        </Row>
        <Row label="Owner" icon={UserIcon}>
          <TaskOwner owner={task.owner} />
        </Row>
        <Row label="Origin" icon={Flag01Icon}>
          <OriginBadge createdByType={task.created_by_type} />
        </Row>
        <Row label="Spend" icon={DollarCircleIcon}>
          <span className="text-zinc-700">{formatSpend(task.spend_total)}</span>
        </Row>
        <Row label="When" icon={Clock01Icon}>
          <span className="text-zinc-700">{formatElapsed(task)}</span>
        </Row>
      </Card>

      {runs.length > 0 ? (
        <Card title="Linked runs">
          <ul className="flex flex-col gap-2" aria-label="Linked runs">
            {runs.map((run) => (
              <li key={run.execution_id}>
                <LinkedRun run={run} />
              </li>
            ))}
          </ul>
        </Card>
      ) : null}

      {isOpenTask(task) ? (
        <Button
          variant="primary"
          size="small"
          className={`w-full ${ACTION_BUTTON_CLASS}`}
          loading={isCancelling}
          onClick={onCancel}
        >
          Cancel task
        </Button>
      ) : null}
    </aside>
  );
}

function Card({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3 rounded-2xl bg-white p-4 ring-[0.5px] ring-zinc-200">
      <Text variant="small" className="font-medium text-zinc-900">
        {title}
      </Text>
      {children}
    </section>
  );
}

function Row({
  label,
  icon,
  children,
}: {
  label: string;
  icon: IconSvgElement;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 text-[13px]">
      <span className="flex items-center gap-1.5 text-zinc-500">
        <Icon icon={icon} size={14} className="text-zinc-400" />
        {label}
      </span>
      {children}
    </div>
  );
}

/** A null owner means Autopilot did the work itself, which the copilot thread
 *  header also marks with the AutoGPT logo rather than a generated avatar. */
function TaskOwner({ owner }: { owner: DelegatedTask["owner"] }) {
  if (!owner) {
    return (
      <span className="flex items-center gap-1.5 text-zinc-700">
        <span className="flex size-5 items-center justify-center rounded-full bg-zinc-100 ring-1 ring-inset ring-zinc-200">
          <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-3" />
        </span>
        Autopilot
      </span>
    );
  }

  return (
    <Link
      href={`/team/${owner.id}`}
      className="flex min-w-0 items-center gap-1.5 text-zinc-700 hover:text-zinc-950"
    >
      <ExpertAvatar name={owner.name} avatarUrl={owner.avatar_url} size={20} />
      <span className="truncate">{owner.name}</span>
    </Link>
  );
}

function LinkedRun({ run }: { run: TaskRunRef }) {
  const body = (
    <div className="flex items-center gap-2 rounded-xl bg-zinc-50 p-2.5 transition-colors group-hover:bg-zinc-100">
      <Icon
        icon={PlayCircleIcon}
        size={16}
        className="shrink-0 text-zinc-400"
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium text-zinc-900">
          {run.agent_name}
        </p>
        <div className="mt-1">
          <RunStatusBadge status={run.status} />
        </div>
      </div>
    </div>
  );

  if (!run.link) return body;

  return (
    <Link
      href={run.link}
      aria-label={`Open run ${run.agent_name}`}
      className="group block"
    >
      {body}
    </Link>
  );
}

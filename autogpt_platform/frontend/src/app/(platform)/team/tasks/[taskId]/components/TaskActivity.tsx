"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { TaskAmendment } from "@/app/api/__generated__/models/taskAmendment";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import Link from "next/link";
import { UserIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  getSessionLink,
  getTaskOriginEntries,
  getTimelineEvents,
  getTimelineLabel,
  type TaskActor,
} from "../../../task-helpers";

interface Props {
  task: DelegatedTask;
}

export function TaskActivity({ task }: Props) {
  const entries = [
    ...getTaskOriginEntries(task).map((origin) => ({
      key: origin.label,
      at: task.created_at,
      actor: origin.actor,
      href: getSessionLink(task.origin_session_id),
      body: <span className="text-zinc-600">{origin.label}</span>,
    })),
    ...getTimelineEvents(task).map((event, index) => ({
      key: `event-${index}`,
      at: event.at,
      actor: toActor(task.owner),
      // An escalation carries the thread its answer is delivered to;
      // everything else falls back to where the task started.
      href: getSessionLink(event.session_id ?? task.origin_session_id),
      body: (
        <>
          <span className="font-medium text-zinc-900">
            {getTimelineLabel(event.kind)}
          </span>
          <span className="text-zinc-600">{event.question ?? event.note}</span>
        </>
      ),
    })),
  ];

  return (
    <section className="flex flex-col gap-3">
      <Text variant="small" className="font-medium text-zinc-900">
        Activity
      </Text>

      <ul className="flex flex-col" aria-label="Activity">
        {entries.map((entry, index) => (
          <Entry
            key={entry.key}
            at={entry.at}
            actor={entry.actor}
            href={entry.href}
            isLast={index === entries.length - 1}
          >
            {entry.body}
          </Entry>
        ))}
      </ul>
    </section>
  );
}

function toActor(owner: DelegatedTask["owner"]): TaskActor {
  return owner
    ? { kind: "expert", name: owner.name, avatarUrl: owner.avatar_url }
    : { kind: "autopilot" };
}

interface EntryProps {
  at: TaskAmendment["at"];
  actor: TaskActor;
  href?: string;
  isLast: boolean;
  children: React.ReactNode;
}

/** One stop on the chain: avatar node on the left rail, a hairline dropping
 *  to the next stop, and the event text beside it — mirrors the copilot
 *  ToolChain so delegated work reads the same everywhere. */
function Entry({ at, actor, href, isLast, children }: EntryProps) {
  const body = (
    <>
      <span className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2">
        {children}
      </span>
      <time className="shrink-0 text-[11px] text-zinc-400">
        {new Date(at).toLocaleString(undefined, {
          month: "short",
          day: "numeric",
          hour: "numeric",
          minute: "2-digit",
        })}
      </time>
    </>
  );

  const rowClass =
    "flex min-h-6 items-center gap-2.5 rounded-xl px-2 py-1 text-[13px]";

  return (
    <li className="flex items-stretch gap-1">
      <div className="flex w-6 flex-col items-center pt-1">
        <EntryAvatar actor={actor} />
        {!isLast && <div className="mt-1 w-px flex-1 bg-zinc-200" />}
      </div>
      <div className={`min-w-0 flex-1 ${isLast ? "pb-0" : "pb-4"}`}>
        {href ? (
          <Link
            href={href}
            className={`${rowClass} transition-colors hover:bg-zinc-100`}
          >
            {body}
          </Link>
        ) : (
          <div className={rowClass}>{body}</div>
        )}
      </div>
    </li>
  );
}

/** Autopilot's own work is stamped with the AutoGPT mark, the same way the
 *  copilot thread header and the task board mark an ownerless task. */
function EntryAvatar({ actor }: { actor: TaskActor }) {
  if (actor.kind === "expert") {
    return (
      <ExpertAvatar name={actor.name} avatarUrl={actor.avatarUrl} size={24} />
    );
  }

  return (
    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-zinc-100 ring-1 ring-inset ring-zinc-200">
      {actor.kind === "autopilot" ? (
        <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-3.5" />
      ) : (
        <Icon icon={UserIcon} size={13} className="text-zinc-500" />
      )}
    </span>
  );
}

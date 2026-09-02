"use client";

import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Text } from "@/components/atoms/Text/Text";
import {
  ArrowDataTransferHorizontalIcon,
  CheckmarkCircle02Icon,
  Comment01Icon,
  HelpCircleIcon,
  PencilEdit02Icon,
  RefreshIcon,
  SentIcon,
  StickyNote02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import {
  getActorName,
  getSessionLink,
  getTaskOriginEntries,
  getTimelineActor,
  getTimelineEvents,
  getTimelineVerb,
  type TaskActor,
} from "../../../../task-helpers";
import { TaskOutcomeReview } from "../TaskOutcomeReview/TaskOutcomeReview";
import { ActivityAnswer } from "./components/ActivityAnswer";
import { ActivityEntry } from "./components/ActivityEntry";
import { ActivityQuote } from "./components/ActivityQuote";

interface Props {
  task: DelegatedTask;
}

interface Badge {
  icon: IconSvgElement;
  className: string;
}

/** Rail badges keyed by event kind, so the feed scans by shape and color
 *  before it's read: blue = moved, amber = blocked, emerald = finished. */
const EVENT_BADGES: Record<string, Badge> = {
  handoff: {
    icon: ArrowDataTransferHorizontalIcon,
    className: "bg-blue-100 text-blue-600",
  },
  escalation: {
    icon: HelpCircleIcon,
    className: "bg-amber-100 text-amber-600",
  },
  answer: { icon: Comment01Icon, className: "bg-violet-100 text-violet-600" },
  note: { icon: StickyNote02Icon, className: "bg-zinc-100 text-zinc-500" },
  retry: { icon: RefreshIcon, className: "bg-zinc-100 text-zinc-500" },
  revision: {
    icon: PencilEdit02Icon,
    className: "bg-rose-100 text-rose-600",
  },
};

const ORIGIN_BADGE: Badge = {
  icon: SentIcon,
  className: "bg-indigo-100 text-indigo-600",
};

const OUTCOME_BADGE: Badge = {
  icon: CheckmarkCircle02Icon,
  className: "bg-emerald-100 text-emerald-600",
};

/** The task's story as a feed: who did what, when — with the words they
 *  left (notes, questions, the outcome) quoted underneath each stop. */
export function TaskActivity({ task }: Props) {
  const owner = toActor(task.owner);
  const events = getTimelineEvents(task);

  // The open question a WAITING_USER task is parked on — answerable right
  // here instead of routing through the session or the Home screen. Only
  // the newest user-facing escalation can still be open.
  const answerableIndex =
    task.status === "WAITING_USER"
      ? events.reduce(
          (last, event, index) =>
            event.kind === "escalation" && event.target !== "manager"
              ? index
              : last,
          -1,
        )
      : -1;

  const originEntries = getTaskOriginEntries(task).map((origin) => ({
    key: origin.label,
    at: task.created_at,
    actor: origin.actor,
    badge: ORIGIN_BADGE,
    href: getSessionLink(task.origin_session_id),
    header: <span className="text-zinc-600">{origin.label}</span>,
    body: null as React.ReactNode,
  }));

  const eventEntries = events.map((event, index) => {
    const actor = getTimelineActor(event, task.owner);
    const quoted = event.question ?? event.note;
    return {
      key: `event-${index}`,
      at: event.at,
      actor,
      badge: EVENT_BADGES[event.kind ?? "note"] ?? EVENT_BADGES.note,
      // An escalation carries the thread its answer is delivered to;
      // everything else falls back to where the task started.
      href: getSessionLink(event.session_id ?? task.origin_session_id),
      header: (
        <>
          <span className="font-semibold text-zinc-900">
            {getActorName(actor)}
          </span>{" "}
          <span className="text-zinc-600">{getTimelineVerb(event)}</span>
        </>
      ),
      body: (
        <>
          {quoted ? <ActivityQuote text={quoted} /> : null}
          {index === answerableIndex ? (
            <ActivityAnswer taskId={task.id} options={event.options ?? []} />
          ) : null}
        </>
      ),
    };
  });

  // The outcome reads in story order, not always last: on a DONE task it is
  // the newest thing that happened, but while a revision is in flight the
  // quoted outcome predates the change request it triggered.
  const outcomeEntries = task.outcome_summary
    ? [
        {
          key: "outcome",
          at: task.updated_at,
          actor: owner,
          badge: OUTCOME_BADGE,
          href: undefined,
          header: (
            <>
              <span className="font-semibold text-zinc-900">
                {getActorName(owner)}
              </span>{" "}
              <span className="text-zinc-600">completed this task</span>
            </>
          ),
          body: (
            <>
              <ActivityQuote text={task.outcome_summary} />
              <div className="mt-2.5">
                <TaskOutcomeReview task={task} />
              </div>
            </>
          ),
        },
      ]
    : [];

  const lastRevisionIndex = events.reduce(
    (last, event, index) => (event.kind === "revision" ? index : last),
    -1,
  );
  const outcomePosition =
    task.status === "DONE" || lastRevisionIndex === -1
      ? eventEntries.length
      : lastRevisionIndex;

  const entries = [
    ...originEntries,
    ...eventEntries.slice(0, outcomePosition),
    ...outcomeEntries,
    ...eventEntries.slice(outcomePosition),
  ];

  return (
    <section className="flex flex-col gap-3">
      <Text variant="small" className="font-medium text-zinc-900">
        Activity
      </Text>

      <ul className="flex flex-col" aria-label="Activity">
        {entries.map((entry, index) => (
          <ActivityEntry
            key={entry.key}
            at={entry.at}
            actor={entry.actor}
            icon={entry.badge.icon}
            iconClassName={entry.badge.className}
            href={entry.href}
            header={entry.header}
            isLast={index === entries.length - 1}
          >
            {entry.body}
          </ActivityEntry>
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

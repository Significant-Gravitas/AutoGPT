"use client";

import { formatUntil } from "@/app/(platform)/home/components/NowNext/helpers";
import { ownerOf } from "@/app/(platform)/tasks/helpers";
import type { Task } from "@/app/api/__generated__/models/task";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { formatTimeAgo } from "@/lib/utils/time";
import { Calendar03Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BriefWords } from "../EmptySession/components/HomeToday/BriefWords";
import { metaOf } from "../EmptySession/components/HomeToday/helpers";
import { useHomeToday } from "../EmptySession/components/HomeToday/useHomeToday";
import { useWeekStats } from "../EmptySession/components/HomeToday/useWeekStats";
import { span } from "../EmptySession/components/HomeToday/weekStats";
import { BlockNote, BlockRow, NeedsYouTag } from "./blocks";

const NEEDS = "brief-needs";
const READY = "brief-ready";
const WORKING = "brief-working";
const RAN = "brief-ran";
const NEXT = "brief-next";

/** The day's brief beside Home. Otto's own opening leads when the morning
 *  job wrote one, followed by where the tasks stand in words; then a row of
 *  counts that jump to their block, and the blocks themselves: what needs
 *  you, what is ready to read, what the team ran, what is in progress, what
 *  comes next, and the week in figures. A block with nothing is left out. */
export function BriefTab() {
  const today = useHomeToday();
  const stats = useWeekStats(today.now);
  const narrative = today.briefing?.narrative;
  const outcomes = today.briefing?.outcomes ?? [];
  const writtenAt =
    narrative && today.briefing
      ? new Date(today.briefing.generated_at).toLocaleTimeString(undefined, {
          hour: "numeric",
          minute: "2-digit",
        })
      : null;

  function taskRows(tasks: Task[], line: (task: Task) => string, tag?: string) {
    return tasks.map((task) => (
      <BlockRow
        key={task.id}
        tile={<Face task={task} today={today} />}
        title={task.title}
        subtitle={line(task)}
        wrap
        trailing={
          tag ? (
            <NeedsYouTag>{tag}</NeedsYouTag>
          ) : (
            <InfoMeta>{metaOf(task, today.now)}</InfoMeta>
          )
        }
        onClick={() => today.openTask(task)}
      />
    ));
  }
  const nameOf = (task: Task) => ownerOf(task, today.experts).name;

  return (
    <div aria-label="Brief" className="flex flex-col gap-5">
      <header className="flex items-center gap-2.5 px-1">
        <ExpertAvatar
          name={AUTOPILOT_NAME}
          avatarUrl={AUTOPILOT_AVATAR_URL}
          size={32}
          shape="soft"
        />
        <div className="flex min-w-0 flex-col">
          <Text variant="inbox-medium" as="h2" tone="ink">
            {new Date(today.now).toLocaleDateString(undefined, {
              weekday: "long",
              day: "numeric",
              month: "long",
            })}
          </Text>
          <Text variant="small" as="span" tone="quiet">
            {writtenAt
              ? `Written by ${AUTOPILOT_NAME} at ${writtenAt}`
              : `From ${AUTOPILOT_NAME}, as things stand now`}
          </Text>
        </div>
      </header>

      <BlockNote className="gap-2.5">
        {narrative ? (
          <p className="text-body leading-6 text-pretty text-foreground-ink">
            {narrative}
          </p>
        ) : null}
        <BriefWords
          today={today}
          stats={stats}
          isFollowUp={Boolean(narrative)}
        />
      </BlockNote>

      <nav aria-label="In this brief" className="flex flex-wrap gap-1.5">
        <Jump to={NEEDS} count={today.needs.length} isWaiting>
          need you
        </Jump>
        <Jump to={READY} count={today.ready.length}>
          ready to read
        </Jump>
        <Jump to={RAN} count={outcomes.length}>
          ran since yesterday
        </Jump>
        <Jump to={WORKING} count={today.working.length}>
          in progress
        </Jump>
        <Jump to={NEXT} count={today.upcoming.length}>
          coming up
        </Jump>
      </nav>

      <Block id={NEEDS} label="Needs you" count={today.needs.length}>
        {taskRows(
          today.needs,
          (task) =>
            task.status_text ?? `${nameOf(task)} · ${metaOf(task, today.now)}`,
          "Answer",
        )}
      </Block>

      <Block id={READY} label="Ready to read" count={today.ready.length}>
        {taskRows(today.ready, (task) => task.result ?? nameOf(task))}
      </Block>

      <Block id={RAN} label="What your team ran" count={outcomes.length}>
        {outcomes.map((outcome) => (
          <BlockRow
            key={outcome.id}
            tile={
              <span
                className={cn(
                  "size-2 rounded-full",
                  outcome.status === "failed" ? "bg-error" : "bg-status-ok",
                )}
              />
            }
            title={outcome.title || outcome.agent_name}
            subtitle={outcome.summary || outcome.expert?.name}
            wrap
            trailing={
              outcome.occurred_at ? (
                <InfoMeta>
                  {formatTimeAgo(String(outcome.occurred_at))}
                </InfoMeta>
              ) : null
            }
            href={outcome.link ?? undefined}
          />
        ))}
      </Block>

      <Block id={WORKING} label="In progress" count={today.working.length}>
        {taskRows(today.working, (task) => task.status_text ?? nameOf(task))}
      </Block>

      <Block id={NEXT} label="Coming up" count={today.upcoming.length}>
        {today.upcoming.map((item) => (
          <BlockRow
            key={item.id}
            tile={<Icon icon={Calendar03Icon} size={14} tone="inherit" />}
            title={item.title}
            subtitle={item.expert?.name}
            trailing={
              <InfoMeta>
                {formatUntil(item.next_run_time, new Date(today.now))}
              </InfoMeta>
            }
          />
        ))}
      </Block>

      <InfoGroup label="This week">
        <BlockNote>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-2.5">
            <Figure term="Tasks finished" value={String(stats.doneWeek)} />
            <Figure term="Work handed off" value={span(stats.workSeconds)} />
            <Figure term="A task takes about" value={span(stats.avgSeconds)} />
            <Figure term="Chats" value={String(stats.chatsWeek)} />
          </dl>
        </BlockNote>
      </InfoGroup>
    </div>
  );
}

interface FaceProps {
  task: Task;
  today: ReturnType<typeof useHomeToday>;
}

function Face({ task, today }: FaceProps) {
  const owner = ownerOf(task, today.experts);
  return (
    <ExpertAvatar name={owner.name} avatarUrl={owner.avatarUrl} size={26} />
  );
}

interface BlockProps {
  id: string;
  label: string;
  count: number;
  children: ReactNode;
}

function Block({ id, label, count, children }: BlockProps) {
  if (count === 0) return null;
  return (
    <div id={id} className="scroll-mt-3">
      <InfoGroup label={label} count={count}>
        <InfoList>{children}</InfoList>
      </InfoGroup>
    </div>
  );
}

interface JumpProps {
  to: string;
  count: number;
  isWaiting?: boolean;
  children: string;
}

/** A count that scrolls the brief to its block; nothing when it is zero. */
function Jump({ to, count, isWaiting, children }: JumpProps) {
  if (count === 0) return null;

  function jump() {
    const isStill = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    document.getElementById(to)?.scrollIntoView({
      behavior: isStill ? "auto" : "smooth",
      block: "start",
    });
  }

  return (
    <button
      type="button"
      onClick={jump}
      className="flex h-7 items-center gap-1.5 rounded-full bg-surface px-2.5 text-body-sm shadow-(--inbox-shadow-card) transition-shadow duration-150 ease-out outline-none hover:shadow-(--inbox-shadow-card-hover) focus-visible:ring-2 focus-visible:ring-focus-inbox"
    >
      <span
        className={cn(
          "font-medium tabular-nums",
          isWaiting ? "text-status-waiting" : "text-foreground-ink",
        )}
      >
        {count}
      </span>
      <span className="text-foreground-text">{children}</span>
    </button>
  );
}

interface FigureProps {
  term: string;
  value: string;
}

function Figure({ term, value }: FigureProps) {
  return (
    <div className="flex min-w-0 flex-col">
      <dt className="truncate text-caption text-foreground-quiet">{term}</dt>
      <dd className="text-body-lg font-medium text-foreground-ink tabular-nums">
        {value}
      </dd>
    </div>
  );
}

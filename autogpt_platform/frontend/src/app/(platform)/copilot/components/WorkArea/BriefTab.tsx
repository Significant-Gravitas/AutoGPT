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
import { InfoMeta } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BriefWords } from "../EmptySession/components/HomeToday/BriefWords";
import { metaOf } from "../EmptySession/components/HomeToday/helpers";
import { useHomeToday } from "../EmptySession/components/HomeToday/useHomeToday";
import { useWeekStats } from "../EmptySession/components/HomeToday/useWeekStats";
import { BlockNote, BlockRow } from "./blocks";
import {
  Block,
  FAILED,
  Jump,
  NEEDS,
  NEXT,
  RAN,
  READY,
  WORKING,
} from "./BriefParts";
import { BriefTaskBlocks } from "./BriefTaskBlocks";
import { storedBriefTasks } from "./briefTasks";
import { StartingToday } from "./StartingToday";
import { SuggestedNext } from "./SuggestedNext/SuggestedNext";
import { WeekFigures } from "./WeekFigures";

/** The day's brief beside Home. Otto's own opening leads when the morning
 *  job wrote one, followed by where the tasks stand in words; then the next
 *  tasks Otto drafted, a row of counts that jump to their block, and the
 *  blocks themselves: what needs you, what finished or failed, what the team
 *  ran, what is in progress, what comes next, and the week in figures. A
 *  block with nothing is left out. */
export function BriefTab() {
  const today = useHomeToday();
  const stats = useWeekStats(today.now);
  const narrative = today.briefing?.narrative;
  const outcomes = today.briefing?.outcomes ?? [];
  const stored = storedBriefTasks(today.briefing?.tasks, today.tasks);
  const writtenAt =
    narrative && today.briefing
      ? new Date(today.briefing.generated_at).toLocaleTimeString(undefined, {
          hour: "numeric",
          minute: "2-digit",
        })
      : null;

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

      <SuggestedNext experts={today.experts} onOpen={today.openTask} />

      <nav aria-label="In this brief" className="flex flex-wrap gap-1.5">
        <Jump
          to={NEEDS}
          count={stored ? stored.waiting.length : today.needs.length}
          isWaiting
        >
          need you
        </Jump>
        <Jump
          to={READY}
          count={stored ? stored.finished.length : today.ready.length}
        >
          {stored ? "finished" : "ready to read"}
        </Jump>
        <Jump to={FAILED} count={stored?.failed.length ?? 0}>
          {"didn't finish"}
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

      <BriefTaskBlocks today={today} stored={stored} />

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

      <StartingToday
        now={today.now}
        experts={today.experts}
        onOpen={today.openTask}
      />

      <Block id={WORKING} label="In progress" count={today.working.length}>
        {today.working.map((task) => (
          <BlockRow
            key={task.id}
            tile={<Face task={task} today={today} />}
            title={task.title}
            subtitle={task.status_text ?? ownerOf(task, today.experts).name}
            wrap
            trailing={<InfoMeta>{metaOf(task, today.now)}</InfoMeta>}
            onClick={() => today.openTask(task)}
          />
        ))}
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

      <WeekFigures stats={stats} />
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

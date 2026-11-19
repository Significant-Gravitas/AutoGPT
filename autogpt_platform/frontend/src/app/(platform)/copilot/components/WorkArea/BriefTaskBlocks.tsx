"use client";

import { ownerOf } from "@/app/(platform)/tasks/helpers";
import type { BriefingTaskItem } from "@/app/api/__generated__/models/briefingTaskItem";
import type { Task } from "@/app/api/__generated__/models/task";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { InfoMeta } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { metaOf } from "../EmptySession/components/HomeToday/helpers";
import type { useHomeToday } from "../EmptySession/components/HomeToday/useHomeToday";
import { BlockRow, NeedsYouTag } from "./blocks";
import { Block, FAILED, NEEDS, READY } from "./BriefParts";
import type { StoredBriefTasks } from "./briefTasks";

type Today = ReturnType<typeof useHomeToday>;

interface Props {
  today: Today;
  stored: StoredBriefTasks | null;
}

/** What waits on the user, what finished and what failed: as this morning's
 *  brief wrote it down, or, for a brief from before it covered tasks, joined
 *  in from the task list as things stand now. */
export function BriefTaskBlocks({ today, stored }: Props) {
  if (!stored) return <LiveTaskBlocks today={today} />;

  function rows(items: BriefingTaskItem[], tag?: string) {
    return items.map((item) => (
      <BlockRow
        key={item.task_id}
        tile={
          <ExpertAvatar
            name={item.owner_name ?? AUTOPILOT_NAME}
            avatarUrl={
              item.owner_name
                ? (item.owner_avatar_url ?? null)
                : AUTOPILOT_AVATAR_URL
            }
            size={26}
          />
        }
        title={item.title}
        subtitle={item.line || item.owner_name}
        wrap
        trailing={tag ? <NeedsYouTag>{tag}</NeedsYouTag> : null}
        onClick={() => today.openTaskByID(item.task_id, item.title)}
      />
    ));
  }

  return (
    <>
      <Block id={NEEDS} label="Waiting on you" count={stored.waiting.length}>
        {rows(stored.waiting, "Answer")}
      </Block>
      <Block
        id={READY}
        label="Finished since your last brief"
        count={stored.finished.length}
      >
        {rows(stored.finished)}
      </Block>
      <Block id={FAILED} label="Didn't finish" count={stored.failed.length}>
        {rows(stored.failed)}
      </Block>
    </>
  );
}

function LiveTaskBlocks({ today }: { today: Today }) {
  const nameOf = (task: Task) => ownerOf(task, today.experts).name;

  function rows(tasks: Task[], line: (task: Task) => string, tag?: string) {
    return tasks.map((task) => {
      const owner = ownerOf(task, today.experts);
      return (
        <BlockRow
          key={task.id}
          tile={
            <ExpertAvatar
              name={owner.name}
              avatarUrl={owner.avatarUrl}
              size={26}
            />
          }
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
      );
    });
  }

  return (
    <>
      <Block id={NEEDS} label="Needs you" count={today.needs.length}>
        {rows(
          today.needs,
          (task) =>
            task.status_text ?? `${nameOf(task)} · ${metaOf(task, today.now)}`,
          "Answer",
        )}
      </Block>
      <Block id={READY} label="Ready to read" count={today.ready.length}>
        {rows(today.ready, (task) => task.result ?? nameOf(task))}
      </Block>
    </>
  );
}

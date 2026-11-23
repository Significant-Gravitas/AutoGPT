"use client";

import { span } from "../EmptySession/components/HomeToday/weekStats";
import type { useWeekStats } from "../EmptySession/components/HomeToday/useWeekStats";
import { InfoGroup } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockNote } from "./blocks";
import { Figure } from "./BriefParts";

interface Props {
  stats: ReturnType<typeof useWeekStats>;
}

/** The week in four figures, each only once there is something to count;
 *  with nothing yet (a first day), the card stays away. */
export function WeekFigures({ stats }: Props) {
  const figures = [
    { term: "Tasks finished", value: stats.doneWeek, show: String },
    { term: "Work handed off", value: stats.workSeconds, show: span },
    { term: "A task takes about", value: stats.avgSeconds, show: span },
    { term: "Chats", value: stats.chatsWeek, show: String },
  ].filter((figure) => figure.value > 0);
  if (figures.length === 0) return null;

  return (
    <InfoGroup label="This week">
      <BlockNote>
        <dl className="grid grid-cols-2 gap-x-3 gap-y-2.5">
          {figures.map(({ term, value, show }) => (
            <Figure key={term} term={term} value={show(value)} />
          ))}
        </dl>
      </BlockNote>
    </InfoGroup>
  );
}

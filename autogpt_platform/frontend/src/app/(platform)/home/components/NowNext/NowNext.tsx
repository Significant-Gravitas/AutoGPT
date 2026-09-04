import { Calendar03Icon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { HomeSectionLabel } from "../HomeSectionLabel/HomeSectionLabel";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { HomeTile } from "../HomeTile/HomeTile";
import { ActiveRow } from "./components/ActiveRow";
import { UpcomingRow } from "./components/UpcomingRow";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

// The rail runs through the centre of each row's 36px marker, which sits
// 16px in from the tile edge.
const TIMELINE_CLASS =
  "relative before:absolute before:bottom-5 before:left-[33px] before:top-5 before:w-px before:bg-zinc-200";

export function NowNext({ dashboard, className }: Props) {
  return (
    <HomeTile className={className} icon={Calendar03Icon} title="Now & next">
      {dashboard.active_tasks.length > 0 ? (
        <div className="border-b border-zinc-100 pb-2">
          <HomeSectionLabel>Working now</HomeSectionLabel>
          <div className={TIMELINE_CLASS}>
            {dashboard.active_tasks.map((item) => (
              <ActiveRow key={item.id} item={item} />
            ))}
          </div>
        </div>
      ) : null}

      <div className="pb-2">
        <HomeSectionLabel>Coming up</HomeSectionLabel>
        {dashboard.upcoming_tasks.length === 0 ? (
          <HomeTileEmpty
            title="Nothing is scheduled"
            description="Your agents are ready when you are."
            className="min-h-0 gap-4 py-6"
          />
        ) : (
          <div className={TIMELINE_CLASS}>
            {dashboard.upcoming_tasks.map((item) => (
              <UpcomingRow key={item.id} item={item} />
            ))}
          </div>
        )}
      </div>
    </HomeTile>
  );
}

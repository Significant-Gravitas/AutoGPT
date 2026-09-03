import { Calendar03Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { HomeTile } from "../HomeTile/HomeTile";
import { ActiveRow } from "./components/ActiveRow";
import { UpcomingRow } from "./components/UpcomingRow";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

// The rail runs through the centre of each row's 24px marker, which sits
// 16px in from the tile edge.
const TIMELINE_CLASS =
  "relative before:absolute before:bottom-4 before:left-[27px] before:top-4 before:w-px before:bg-zinc-200";

export function NowNext({ dashboard, className }: Props) {
  return (
    <HomeTile className={className} icon={Calendar03Icon} title="Now & next">
      {dashboard.active_tasks.length > 0 ? (
        <div className="border-b border-zinc-100 pb-2">
          <SectionLabel>Working now</SectionLabel>
          <div className={TIMELINE_CLASS}>
            {dashboard.active_tasks.map((item) => (
              <ActiveRow key={item.id} item={item} />
            ))}
          </div>
        </div>
      ) : null}

      <div className="pb-2">
        <SectionLabel>Coming up</SectionLabel>
        {dashboard.upcoming_tasks.length === 0 ? (
          <HomeTileEmpty
            icon={Clock01Icon}
            title="Nothing is scheduled"
            description="Your agents are ready when you are."
            className="min-h-0 py-6"
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

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <Text
      variant="small-medium"
      className="px-4 pb-1 pt-3 text-[11px] uppercase tracking-[0.06em] text-zinc-400"
    >
      {children}
    </Text>
  );
}

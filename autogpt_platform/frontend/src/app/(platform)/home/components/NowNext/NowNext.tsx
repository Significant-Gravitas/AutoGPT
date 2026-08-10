import { Calendar03Icon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTile } from "../HomeTile/HomeTile";
import { ActiveRow } from "./components/ActiveRow";
import { UpcomingRow } from "./components/UpcomingRow";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function NowNext({ dashboard, className }: Props) {
  return (
    <HomeTile
      className={className}
      contentClassName="flex flex-col gap-4"
      title={
        <div className="flex items-center gap-2">
          <Icon
            icon={Calendar03Icon}
            size={18}
            className="text-zinc-500"
            aria-hidden="true"
          />
          <Text variant="h5" className="text-zinc-950">
            Now &amp; next
          </Text>
        </div>
      }
      header={
        <Text variant="large" className="text-zinc-600">
          Live work and the next scheduled handoffs.
        </Text>
      }
    >
      {dashboard.active_tasks.length > 0 ? (
        <div>
          <Text variant="small" className="font-medium text-zinc-500">
            Working now
          </Text>
          <div className="relative mt-1 before:absolute before:bottom-4 before:left-4 before:top-4 before:w-px before:bg-zinc-200">
            {dashboard.active_tasks.map((item) => (
              <ActiveRow key={item.id} item={item} />
            ))}
          </div>
        </div>
      ) : null}

      <div>
        <Text variant="small" className="font-medium text-zinc-500">
          Coming up
        </Text>
        {dashboard.upcoming_tasks.length === 0 ? (
          <div className="py-6 text-center">
            <Text variant="small" className="text-pretty text-zinc-500">
              Nothing is scheduled. Your agents are ready when you are.
            </Text>
          </div>
        ) : (
          <div className="relative mt-1 before:absolute before:bottom-4 before:left-4 before:top-4 before:w-px before:bg-zinc-200">
            {dashboard.upcoming_tasks.map((item) => (
              <UpcomingRow key={item.id} item={item} />
            ))}
          </div>
        )}
      </div>
    </HomeTile>
  );
}

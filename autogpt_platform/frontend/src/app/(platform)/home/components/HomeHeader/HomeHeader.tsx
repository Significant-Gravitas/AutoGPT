import { Calendar03Icon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatHeaderDate, getHomeStatusLine } from "../../helpers";

interface Props {
  greeting: string;
  name: string;
  dashboard: HomeDashboardResponse;
}

export function HomeHeader({ greeting, name, dashboard }: Props) {
  const date = formatHeaderDate(dashboard.generated_at, dashboard.timezone);
  const status = getHomeStatusLine(dashboard);

  return (
    <header className="my-6 flex items-start justify-between gap-6 px-1">
      <div className="min-w-0">
        <Text
          variant="h4"
          className="text-pretty tracking-[-0.025em] text-zinc-950"
        >
          {greeting},{" "}
          <span className="bg-gradient-to-r from-purple-500 to-purple-300 bg-clip-text text-transparent">
            {name}
          </span>
        </Text>
        <Text variant="large" className="mt-3 text-pretty text-zinc-500">
          {status.split(/(\d+)/).map((part, index) =>
            /^\d+$/.test(part) ? (
              <strong
                key={`${part}-${index}`}
                className="font-semibold tabular-nums text-zinc-700"
              >
                {part}
              </strong>
            ) : (
              part
            ),
          )}
        </Text>
      </div>
      <time
        dateTime={new Date(dashboard.generated_at).toISOString()}
        className="flex shrink-0 items-center gap-3 text-right"
      >
        <Icon
          icon={Calendar03Icon}
          size={22}
          className="text-zinc-500"
          aria-hidden="true"
        />
        <div>
          <Text variant="large" className="font-medium text-zinc-900">
            {date.weekday}
          </Text>
          <Text variant="body-medium" className="mt-0.5 text-zinc-500">
            {date.calendarDate}
          </Text>
        </div>
      </time>
    </header>
  );
}

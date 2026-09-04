import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
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
    <header className="flex items-end justify-between gap-6 px-1 pb-5 pt-1">
      <div className="min-w-0">
        <Text
          variant="large-semibold"
          as="h1"
          className="text-pretty text-[1.25rem] leading-7 tracking-[-0.01em] text-zinc-950"
        >
          {greeting}, {name}
        </Text>
        <Text variant="body" className="mt-0.5 text-pretty text-zinc-950">
          {status.split(/(\d+)/).map((part, index) =>
            /^\d+$/.test(part) ? (
              <span
                key={`${part}-${index}`}
                className="font-medium tabular-nums"
              >
                {part}
              </span>
            ) : (
              part
            ),
          )}
        </Text>
      </div>
      <time
        dateTime={new Date(dashboard.generated_at).toISOString()}
        className="shrink-0 text-right"
      >
        <Text variant="large-medium" className="text-zinc-950">
          {date.weekday}
        </Text>
        <Text variant="body" className="text-zinc-950">
          {date.calendarDate}
        </Text>
      </time>
    </header>
  );
}

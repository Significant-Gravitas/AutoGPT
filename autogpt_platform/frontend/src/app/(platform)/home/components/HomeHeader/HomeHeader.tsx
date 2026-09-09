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
          variant="lead-semibold"
          as="h1"
          tone="primary"
          className="text-pretty tracking-[-0.01em]"
        >
          {greeting}, {name}
        </Text>
        <Text variant="body" tone="primary" className="mt-0.5 text-pretty">
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
        <Text variant="large-medium" tone="primary">
          {date.weekday}
        </Text>
        <Text variant="body" tone="primary">
          {date.calendarDate}
        </Text>
      </time>
    </header>
  );
}

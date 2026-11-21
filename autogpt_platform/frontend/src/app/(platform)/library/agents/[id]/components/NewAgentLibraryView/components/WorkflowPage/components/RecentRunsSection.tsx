import type { PageRecentRuns } from "@/app/api/__generated__/models/pageRecentRuns";
import { Text } from "@/components/atoms/Text/Text";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { formatDate } from "@/lib/utils/time";

interface Props {
  section: PageRecentRuns;
  onSelectRun?: (runID: string) => void;
}

export function RecentRunsSection({ section, onSelectRun }: Props) {
  if (!section.runs.length) {
    return (
      <Text variant="body" tone="secondary">
        No runs yet.
      </Text>
    );
  }
  return (
    <ul className="flex flex-col divide-y divide-separator">
      {section.runs.map((run) => {
        const when = run.started_at
          ? formatDate(new Date(run.started_at))
          : "Queued";
        const cost = `$${((run.cost_cents ?? 0) / 100).toFixed(2)}`;
        return (
          <li key={run.id}>
            <button
              type="button"
              disabled={!onSelectRun}
              onClick={() => onSelectRun?.(run.id)}
              className="flex w-full items-center justify-between gap-3 py-2 text-left disabled:cursor-default"
            >
              <span className="flex min-w-0 items-center gap-2">
                <RunStatusBadge status={run.status} />
                <Text variant="body" className="truncate">
                  {when}
                </Text>
              </span>
              <Text variant="body" tone="secondary">
                {cost}
              </Text>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

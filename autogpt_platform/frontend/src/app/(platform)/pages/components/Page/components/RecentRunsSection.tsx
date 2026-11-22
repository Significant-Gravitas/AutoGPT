import type { PageRecentRuns } from "@/app/api/__generated__/models/pageRecentRuns";
import { Text } from "@/components/atoms/Text/Text";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { formatDate } from "@/lib/utils/time";
import NextLink from "next/link";
import { dollars, libraryRunHref } from "../helpers";

interface Props {
  section: PageRecentRuns;
}

export function RecentRunsSection({ section }: Props) {
  if (!section.runs.length) {
    return (
      <Text variant="body" tone="secondary">
        No runs yet.
      </Text>
    );
  }
  return (
    <ul className="flex flex-col divide-y divide-separator">
      {section.runs.map((run) => (
        <li key={run.id}>
          <NextLink
            href={libraryRunHref(section.library_agent_id, run.id)}
            className="flex w-full flex-col gap-1 py-2 hover:bg-surface-muted/60"
          >
            <span className="flex items-center justify-between gap-3">
              <span className="flex min-w-0 items-center gap-2">
                <RunStatusBadge status={run.status} />
                <Text variant="body" className="truncate">
                  {run.started_at
                    ? formatDate(new Date(run.started_at))
                    : "Queued"}
                </Text>
              </span>
              <Text variant="body" tone="secondary" tabular>
                {dollars(run.cost_cents ?? 0)}
              </Text>
            </span>
            {run.summary ? (
              <Text variant="small" tone="secondary" className="line-clamp-2">
                {run.summary}
              </Text>
            ) : null}
          </NextLink>
        </li>
      ))}
    </ul>
  );
}

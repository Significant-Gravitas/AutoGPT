"use client";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { formatSpend } from "./helpers";
import { useTeamSpendSection } from "./useTeamSpendSection";

interface Props {
  orgId: string;
  canManageBilling: boolean;
}

// Spend-by-team breakdown for the org Billing tab. Rendered inside the
// billing-gated area; the extra guard keeps it inert if ever mounted elsewhere.
// Uses the endpoint's own team_name (not TeamBadge) because a billing manager
// can see spend for teams they aren't a member of, which TeamBadge would hide.
export function TeamSpendSection({ orgId, canManageBilling }: Props) {
  const { buckets, isLoading, isError, refetch } = useTeamSpendSection(
    orgId,
    canManageBilling,
  );

  if (!canManageBilling) {
    return null;
  }

  return (
    <section
      className="flex flex-col gap-4"
      data-testid="org-team-spend-section"
    >
      <div className="flex flex-col gap-1">
        <Text variant="h4" as="h2">
          Spend by team
        </Text>
        <Text variant="body" className="text-zinc-500">
          How your organization’s credit usage breaks down across teams.
        </Text>
      </div>

      {isLoading ? (
        <Skeleton className="h-40 w-full" />
      ) : isError ? (
        <ErrorCard
          responseError={{ message: "Failed to load team spend" }}
          context="team spend"
          onRetry={() => refetch()}
        />
      ) : buckets.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          No spend recorded yet.
        </Text>
      ) : (
        <div className="overflow-x-auto rounded-large border border-zinc-200">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-100 bg-zinc-50/50">
                <th className="px-3 py-2 font-medium text-zinc-600">Team</th>
                <th className="px-3 py-2 text-right font-medium text-zinc-600">
                  Spend
                </th>
                <th className="px-3 py-2 text-right font-medium text-zinc-600">
                  Runs
                </th>
              </tr>
            </thead>
            <tbody>
              {buckets.map((bucket) => (
                <tr
                  key={bucket.team_id ?? "org-home"}
                  className="border-b border-zinc-50 last:border-0"
                  data-testid="team-spend-row"
                >
                  <td className="px-3 py-2">
                    {bucket.team_id ? (
                      <Badge variant="info" size="small">
                        {bucket.team_name ?? "Team"}
                      </Badge>
                    ) : (
                      <Text variant="small" as="span" className="text-zinc-600">
                        Organization
                      </Text>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {formatSpend(bucket.total_spent)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-zinc-500">
                    {bucket.transaction_count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

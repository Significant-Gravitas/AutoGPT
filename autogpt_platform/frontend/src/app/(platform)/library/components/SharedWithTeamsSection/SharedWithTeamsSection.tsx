"use client";

import { Text } from "@/components/atoms/Text/Text";
import { TeamBadge } from "@/components/contextual/TeamBadge/TeamBadge";

import { capabilityLabel } from "../ShareAgentDialog/helpers";
import { useSharedWithTeamsSection } from "./useSharedWithTeamsSection";

// Agents other members shared with the viewer's teams. Rides /grants/received.
// Renders nothing for solo users, while loading, or when nothing is shared, so
// it only appears as a distinct library section once there's content to show.
export function SharedWithTeamsSection() {
  const { hasTeams, grants, isLoading, isError } = useSharedWithTeamsSection();

  if (!hasTeams || isLoading || isError || grants.length === 0) return null;

  return (
    <section className="space-y-3" data-testid="shared-with-teams-section">
      <Text variant="h4" as="h2">
        Shared with your teams
      </Text>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {grants.map((grant) => (
          <div
            key={grant.id}
            className="rounded-large border border-zinc-200 bg-white p-4"
            data-testid="shared-with-teams-card"
          >
            <div className="flex items-center justify-between gap-2">
              <Text variant="body-medium" className="min-w-0 truncate">
                {grant.graph_name ?? "Shared agent"}
              </Text>
              <TeamBadge teamId={grant.principal_id} />
            </div>
            {grant.graph_description ? (
              <Text variant="small" className="mt-1 line-clamp-2 text-zinc-500">
                {grant.graph_description}
              </Text>
            ) : null}
            <Text variant="small" className="mt-2 block text-zinc-500">
              {capabilityLabel(grant.capability)} ·{" "}
              {grant.follow_latest
                ? "latest version"
                : `v${grant.agent_graph_version}`}
            </Text>
          </div>
        ))}
      </div>
    </section>
  );
}

"use client";

import { Card } from "@/components/atoms/Card/Card";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Select } from "@/components/atoms/Select/Select";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

import { MessageVolumeChart } from "./MessageVolumeChart";
import { ServerGrowthChart } from "./ServerGrowthChart";
import { SimpleTable } from "./SimpleTable";
import { SummaryCard } from "./SummaryCard";
import { useBotsContent } from "./useBotsContent";
import {
  DAYS_OPTIONS,
  formatDate,
  formatDuration,
  formatNumber,
  formatPercent,
  PLATFORM_OPTIONS,
} from "./helpers";

export function BotsContent() {
  const state = useBotsContent();
  const { summary } = state;

  const tiles = [
    { label: "Live servers", value: formatNumber(summary?.live_servers) },
    {
      label: "Messages received",
      value: formatNumber(summary?.messages_received),
    },
    { label: "Replies sent", value: formatNumber(summary?.replies_sent) },
    { label: "Commands used", value: formatNumber(summary?.commands_used) },
    {
      label: "Error rate",
      value: formatPercent(summary?.error_rate),
      subtitle: `${formatNumber(summary?.stream_errors)} stream errors`,
    },
    {
      label: "Avg reply time",
      value: formatDuration(summary?.avg_reply_ms),
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap gap-3">
        <Select
          id="platform-filter"
          label="Platform"
          hideLabel
          value={state.platform}
          onValueChange={state.setPlatform}
          options={PLATFORM_OPTIONS}
          size="small"
        />
        <Select
          id="days-filter"
          label="Range"
          hideLabel
          value={String(state.days)}
          onValueChange={(value) => state.setDays(Number(value))}
          options={DAYS_OPTIONS}
          size="small"
        />
      </div>

      {state.isError ? (
        <ErrorCard
          httpError={{
            message:
              state.error instanceof Error
                ? state.error.message
                : "Failed to load bot analytics",
          }}
          context="bot analytics"
          hint="Make sure the backend is running the latest build and try again."
        />
      ) : state.isLoading ? (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} className="h-24 rounded-lg" />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
            {tiles.map((tile) => (
              <SummaryCard key={tile.label} {...tile} />
            ))}
          </div>

          <Card>
            <h2 className="mb-4 text-xl font-semibold">
              Server growth & sharding outlook
            </h2>
            <ServerGrowthChart data={state.servers} />
          </Card>

          <Card>
            <h2 className="mb-4 text-xl font-semibold">Message volume</h2>
            <MessageVolumeChart data={state.messages} />
          </Card>

          <Card>
            <h2 className="mb-4 text-xl font-semibold">
              Top servers by activity
            </h2>
            <SimpleTable
              columns={["Server", "Messages", "Commands"]}
              rows={state.topServers.map((server) => [
                server.name || server.server_id,
                formatNumber(server.messages),
                formatNumber(server.commands),
              ])}
            />
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <h2 className="mb-4 text-xl font-semibold">Command usage</h2>
              <SimpleTable
                columns={["Command", "Uses"]}
                rows={state.commands.map((command) => [
                  `/${command.command}`,
                  formatNumber(command.uses),
                ])}
              />
            </Card>

            <Card>
              <h2 className="mb-4 text-xl font-semibold">Server roster</h2>
              <SimpleTable
                columns={["Server", "Status", "Joined"]}
                rows={state.roster.map((guild) => [
                  guild.name || guild.server_id,
                  guild.active ? "Active" : "Left",
                  formatDate(guild.joined_at),
                ])}
              />
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

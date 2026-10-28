"use client";

import { useState } from "react";

import {
  useGetV2BotMessageTimeseries,
  useGetV2BotServerCountTimeseriesShardingCurve,
  useGetV2BotServerRoster,
  useGetV2BotUsageSummary,
  useGetV2CommandUsageBreakdown,
  useGetV2TopServersByActivity,
} from "@/app/api/__generated__/endpoints/admin/admin";
import type { Platform } from "@/app/api/__generated__/models/platform";
import { okData } from "@/app/api/helpers";

export function useBotsContent() {
  const [days, setDays] = useState(30);
  const [platform, setPlatform] = useState("all");

  const platformParam = platform === "all" ? undefined : (platform as Platform);
  const windowParams = {
    days,
    ...(platformParam && { platform: platformParam }),
  };

  // NOTE: the `select: okData` option must be inlined per call — hoisting it to
  // a shared const collapses okData's generic to `any` and loses response types.
  const summary = useGetV2BotUsageSummary(windowParams, {
    query: { select: okData },
  });
  const messages = useGetV2BotMessageTimeseries(windowParams, {
    query: { select: okData },
  });
  const servers = useGetV2BotServerCountTimeseriesShardingCurve(windowParams, {
    query: { select: okData },
  });
  const topServers = useGetV2TopServersByActivity(windowParams, {
    query: { select: okData },
  });
  const commands = useGetV2CommandUsageBreakdown(windowParams, {
    query: { select: okData },
  });
  const roster = useGetV2BotServerRoster(
    platformParam ? { platform: platformParam } : {},
    { query: { select: okData } },
  );

  return {
    days,
    setDays,
    platform,
    setPlatform,
    summary: summary.data,
    messages: messages.data ?? [],
    servers: servers.data ?? [],
    topServers: topServers.data ?? [],
    commands: commands.data ?? [],
    roster: roster.data ?? [],
    isLoading:
      summary.isLoading ||
      messages.isLoading ||
      servers.isLoading ||
      topServers.isLoading ||
      commands.isLoading ||
      roster.isLoading,
    isError:
      summary.isError ||
      messages.isError ||
      servers.isError ||
      topServers.isError ||
      commands.isError ||
      roster.isError,
    error:
      summary.error ||
      messages.error ||
      servers.error ||
      topServers.error ||
      commands.error ||
      roster.error,
  };
}

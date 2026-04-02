"use server";

import BackendApi from "@/lib/autogpt-server-api";
import type {
  PlatformCostDashboard,
  PlatformCostLogsResponse,
} from "@/lib/autogpt-server-api";

export async function getPlatformCostDashboard(params?: {
  start?: string;
  end?: string;
  provider?: string;
  user_id?: string;
}): Promise<PlatformCostDashboard> {
  const api = new BackendApi();
  return api.getPlatformCostDashboard(params);
}

export async function getPlatformCostLogs(params?: {
  start?: string;
  end?: string;
  provider?: string;
  user_id?: string;
  page?: number;
  page_size?: number;
}): Promise<PlatformCostLogsResponse> {
  const api = new BackendApi();
  return api.getPlatformCostLogs(params);
}

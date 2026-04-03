"use server";

import {
  getV2GetPlatformCostDashboard,
  getV2GetPlatformCostLogs,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { toDateOrUndefined } from "./helpers";

export async function getPlatformCostDashboard(params?: {
  start?: string;
  end?: string;
  provider?: string;
  user_id?: string;
}) {
  const response = await getV2GetPlatformCostDashboard({
    start: toDateOrUndefined(params?.start),
    end: toDateOrUndefined(params?.end),
    provider: params?.provider || undefined,
    user_id: params?.user_id || undefined,
  });
  return okData(response);
}

export async function getPlatformCostLogs(params?: {
  start?: string;
  end?: string;
  provider?: string;
  user_id?: string;
  page?: number;
  page_size?: number;
}) {
  const response = await getV2GetPlatformCostLogs({
    start: toDateOrUndefined(params?.start),
    end: toDateOrUndefined(params?.end),
    provider: params?.provider || undefined,
    user_id: params?.user_id || undefined,
    page: params?.page,
    page_size: params?.page_size,
  });
  return okData(response);
}

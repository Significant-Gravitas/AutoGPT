"use server";

import {
  getV2GetPlatformCostDashboard,
  getV2GetPlatformCostLogs,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";

export async function getPlatformCostDashboard(params?: {
  start?: string;
  end?: string;
  provider?: string;
  user_id?: string;
}) {
  const response = await getV2GetPlatformCostDashboard(params);
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
  const response = await getV2GetPlatformCostLogs(params);
  return okData(response);
}

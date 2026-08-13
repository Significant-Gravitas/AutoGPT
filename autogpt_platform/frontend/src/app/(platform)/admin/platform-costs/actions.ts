import {
  getV2GetPlatformCostDashboard,
  getV2GetPlatformCostLogs,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";

// Backend expects ISO datetime strings. The generated client's URL builder
// calls .toString() on values, which for Date objects produces the human
// "Tue Mar 31 2026 22:00:00 GMT+0000 (Coordinated Universal Time)" format
// that FastAPI rejects with 422. We already pass UTC ISO from the URL, so
// forward the raw strings through the `as unknown as Date` cast to match
// the generated typing without triggering Date.toString().
export async function getPlatformCostDashboard(params?: {
  start?: string;
  end?: string;
  provider?: string;
  user_id?: string;
}) {
  const response = await getV2GetPlatformCostDashboard({
    start: (params?.start || undefined) as unknown as Date | undefined,
    end: (params?.end || undefined) as unknown as Date | undefined,
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
    start: (params?.start || undefined) as unknown as Date | undefined,
    end: (params?.end || undefined) as unknown as Date | undefined,
    provider: params?.provider || undefined,
    user_id: params?.user_id || undefined,
    page: params?.page,
    page_size: params?.page_size,
  });
  return okData(response);
}

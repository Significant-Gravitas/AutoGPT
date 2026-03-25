"use server";

import { revalidatePath } from "next/cache";
import BackendApi from "@/lib/autogpt-server-api";
import type { UserRateLimitResponse } from "@/lib/autogpt-server-api/types";

export async function getUserRateLimit(
  userId: string,
): Promise<UserRateLimitResponse> {
  const api = new BackendApi();
  return api.getUserRateLimit(userId);
}

export async function resetUserRateLimit(
  userId: string,
): Promise<UserRateLimitResponse> {
  const api = new BackendApi();
  const result = await api.resetUserRateLimit(userId);
  revalidatePath("/admin/rate-limits");
  return result;
}

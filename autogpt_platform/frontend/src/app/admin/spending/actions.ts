"use server";

import { revalidatePath } from "next/cache";
import BackendApi from "@/lib/autogpt-server-api";
import {
  GrantHistoryResponse,
  NotificationPreferenceDTO,
  StoreListingsWithVersionsResponse,
  StoreSubmissionsResponse,
  SubmissionStatus,
  UserBalancesResponse,
} from "@/lib/autogpt-server-api/types";

export async function addDollars(formData: FormData) {
  const data = {
    user_id: formData.get("id") as string,
    amount: parseInt(formData.get("amount") as string),
    comments: formData.get("comments") as string,
  };
  const api = new BackendApi();
  await api.addUserCredits(data.user_id, data.amount, data.comments);

  revalidatePath("/admin/spending");
}

export async function getUserBalances(
  page: number = 1,
  pageSize: number = 20,
  search?: string,
): Promise<UserBalancesResponse> {
  const data: Record<string, any> = {
    page,
    page_size: pageSize,
  };
  if (search) {
    data.search = search;
  }
  const api = new BackendApi();
  const balances = await api.getUserBalances(data);

  return balances;
}

export async function getGrantHistory(
  page: number = 1,
  pageSize: number = 20,
  search?: string,
): Promise<GrantHistoryResponse> {
  const data: Record<string, any> = {
    page,
    page_size: pageSize,
  };
  if (search) {
    data.search = search;
  }
  const api = new BackendApi();
  const grants = await api.getGrantHistory(data);
  console.log(grants);
  return grants;
}

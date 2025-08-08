"use server";

import { revalidatePath } from "next/cache";
import BackendApi from "@/lib/autogpt-server-api";
import {
  UsersBalanceHistoryResponse,
  CreditTransactionType,
} from "@/lib/autogpt-server-api/types";

export async function addDollars(formData: FormData) {
  const data = {
    user_id: formData.get("id") as string,
    amount: parseInt(formData.get("amount") as string),
    comments: formData.get("comments") as string,
  };
  const api = new BackendApi();
  const resp = await api.addUserCredits(
    data.user_id,
    data.amount,
    data.comments,
  );
  console.log(resp);
  revalidatePath("/admin/spending");
}

export async function getUsersTransactionHistory(
  page: number = 1,
  pageSize: number = 20,
  search?: string,
  transactionType?: CreditTransactionType,
): Promise<UsersBalanceHistoryResponse> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: Record<string, any> = {
    page,
    page_size: pageSize,
  };
  if (search) {
    data.search = search;
  }
  if (transactionType) {
    data.transaction_filter = transactionType;
  }
  const api = new BackendApi();
  const history = await api.getUsersHistory(data);
  return history;
}

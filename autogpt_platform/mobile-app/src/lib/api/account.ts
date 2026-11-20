import { apiRequest } from "./client";

export interface AccountDeletion {
  status: "open" | "scheduled" | "too_late";
  /** Until then, signing in again on the web keeps the account. */
  erase_after: string | null;
}

/** Closes the account, as the web's Settings does; the web keeps the restore step. */
export function closeAccount(email: string) {
  return apiRequest<AccountDeletion>("/api/account/deletion", {
    method: "POST",
    body: { confirm_email: email, source: "settings" },
  });
}

import { apiRequest } from "./client";

export interface AccountDeletion {
  status: "open" | "scheduled" | "too_late";
  /** Until then, signing in on the web and choosing Keep my account restores it. */
  erase_after: string | null;
}

/** Closes the account, as the web's Settings does; restoring it happens only on the web. */
export function closeAccount(email: string) {
  return apiRequest<AccountDeletion>("/api/account/deletion", {
    method: "POST",
    body: { confirm_email: email, source: "settings" },
  });
}

import { TERMS_VERSION } from "@/lib/legal";

import { apiRequest } from "./client";

/** After a sign-up that came back signed in: POST /api/auth/user, then the consent record. */
export async function provisionNewAccount(marketingOptOut: boolean) {
  await apiRequest("/api/auth/user", { method: "POST" });
  try {
    await apiRequest("/api/auth/user/consent", {
      method: "POST",
      body: { terms_version: TERMS_VERSION, marketing_opt_out: marketingOptOut },
    });
  } catch {
    // As on the web, a lost consent write must not undo the sign-up.
  }
}

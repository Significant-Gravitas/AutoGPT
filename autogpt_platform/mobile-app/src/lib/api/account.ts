import { assertSameSession, sessionMark } from "@/lib/auth/token";
import { TERMS_VERSION } from "@/lib/legal";

import { apiRequest } from "./client";

/** After a sign-up that came back signed in: POST /api/auth/user, then the consent record. */
export async function provisionNewAccount(marketingOptOut: boolean) {
  const mark = sessionMark();
  await apiRequest("/api/auth/user", { method: "POST" });
  try {
    // The choice belongs to the account just created, never to whoever signed in since.
    assertSameSession(mark);
    await apiRequest("/api/auth/user/consent", {
      method: "POST",
      body: { terms_version: TERMS_VERSION, marketing_opt_out: marketingOptOut },
    });
  } catch {
    // As on the web, a lost consent write must not undo the sign-up.
  }
}

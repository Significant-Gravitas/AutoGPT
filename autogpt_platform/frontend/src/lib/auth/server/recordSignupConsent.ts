import { postV1RecordUserConsent } from "@/app/api/__generated__/endpoints/auth/auth";
import { TERMS_VERSION } from "@/lib/legal";
import * as Sentry from "@sentry/nextjs";

/**
 * Records terms acceptance, and a marketing refusal if there was one, on an
 * account created in the same request. Never throws: a failed consent write is
 * reported, but it must not fail or roll back the signup.
 */
export async function recordSignupConsent(
  marketingOptOut: boolean,
): Promise<void> {
  try {
    await postV1RecordUserConsent({
      terms_version: TERMS_VERSION,
      marketing_opt_out: marketingOptOut,
    });
  } catch (error) {
    console.error("Failed to record signup consent:", error);
    Sentry.captureException(error, {
      tags: { signup_step: "record_consent" },
    });
  }
}

import { redirect } from "next/navigation";

/**
 * Root page always redirects to /copilot.
 * The /copilot page handles the feature flag check and redirects to /library if needed.
 * This single-check approach avoids race conditions with LaunchDarkly initialization.
 * See: SECRT-1845
 */
export default function Page() {
  redirect("/copilot");
}

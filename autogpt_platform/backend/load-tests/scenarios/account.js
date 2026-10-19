// What every page load asks about the account: credits and the subscription
// (the slowest read in production), onboarding state, integration
// credentials and providers, organisations and the timezone. These are the
// second-largest family of user-facing requests in production and they sit
// in front of every other journey.
import { iterationPhase } from "../configs/profiles.js";
import { api } from "../lib/http.js";

export function accountReads(data, profile) {
  const phase = iterationPhase(profile);
  const reads = [
    ["/credits", "account.credits"],
    ["/credits/subscription", "account.subscription"],
    ["/credits/trial", "account.trial"],
    ["/onboarding", "account.onboarding"],
    ["/onboarding/completed", "account.onboarding_completed"],
    ["/integrations/credentials", "account.credentials"],
    ["/integrations/providers", "account.providers"],
    ["/integrations/providers/system", "account.providers_system"],
    ["/orgs", "account.orgs"],
    ["/auth/user/timezone", "account.timezone"],
  ];
  for (const [path, endpoint] of reads) {
    api("GET", path, { data, endpoint, phase });
  }
}

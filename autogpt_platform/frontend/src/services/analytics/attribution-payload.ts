import type { UserAttributionInput } from "@/app/api/__generated__/models/userAttributionInput";
import { readAccountCreatedFlag } from "@/services/analytics/account-created-cookie";
import {
  getAnonymousID,
  getPostHogDeviceID,
  readFirstLanding,
} from "@/services/analytics/anonymous-id";

/**
 * Everything the browser knows about where this user came from. The DataFast
 * ids travel as request headers (see the API mutator), so they are not here.
 */
export function buildAttributionPayload(): UserAttributionInput {
  const landing = readFirstLanding();
  return {
    anonymous_id: getAnonymousID(),
    posthog_distinct_id: getPostHogDeviceID(),
    landing_path: landing?.path ?? null,
    referrer: landing?.referrer ?? null,
    utm_source: landing?.utm_source ?? null,
    utm_medium: landing?.utm_medium ?? null,
    utm_campaign: landing?.utm_campaign ?? null,
    signup_method: readAccountCreatedFlag(),
  };
}

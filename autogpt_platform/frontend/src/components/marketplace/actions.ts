"use server";

import * as Sentry from "@sentry/nextjs";
import MarketplaceAPI, { AnalyticsEvent } from "@/lib/marketplace-api";

export async function makeAnalyticsEvent(event: AnalyticsEvent) {
  return await Sentry.withServerActionInstrumentation(
    "makeAnalyticsEvent",
    {},
    async () => {
      const apiUrl = process.env.AGPT_SERVER_API_URL;
      const api = new MarketplaceAPI();
      await api.makeAnalyticsEvent(event);
    },
  );
}

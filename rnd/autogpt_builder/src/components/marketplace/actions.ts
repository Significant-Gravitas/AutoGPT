"use server";

import MarketplaceAPI, { AnalyticsEvent } from "@/lib/marketplace-api";

export async function makeAnalyticsEvent(event: AnalyticsEvent) {
  const apiUrl = process.env.AGPT_SERVER_API_URL;
  const api = new MarketplaceAPI();
  await api.makeAnalyticsEvent(event);
}

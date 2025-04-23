// "use server";

// import * as Sentry from "@sentry/nextjs";
// import MarketplaceAPI, { AnalyticsEvent } from "@/lib/marketplace-api";
// import { checkAuth } from "@/lib/supabase/server";

// export async function makeAnalyticsEvent(event: AnalyticsEvent) {
//   return await Sentry.withServerActionInstrumentation(
//     "makeAnalyticsEvent",
//     {},
//     async () => {
//       await checkAuth();
//       const apiUrl = process.env.AGPT_SERVER_API_URL;
//       const api = new MarketplaceAPI();
//       await api.makeAnalyticsEvent(event);
//     },
//   );
// }

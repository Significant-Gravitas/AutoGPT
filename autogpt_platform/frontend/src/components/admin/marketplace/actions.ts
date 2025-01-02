// "use server";
// import MarketplaceAPI from "@/lib/marketplace-api";
// import ServerSideMarketplaceAPI from "@/lib/marketplace-api/server-client";
// import { revalidatePath } from "next/cache";
// import * as Sentry from "@sentry/nextjs";
// import { redirect } from "next/navigation";

// export async function checkAuth() {
//     const supabase = getServerSupabase();
//     if (!supabase) {
//       console.error("No supabase client");
//       redirect("/login");
//     }
//     const { data, error } = await supabase.auth.getUser();
//     if (error || !data?.user) {
//       redirect("/login");
//     }
//   }

// export async function approveAgent(
//   agentId: string,
//   version: number,
//   comment: string,
// ) {
//   return await Sentry.withServerActionInstrumentation(
//     "approveAgent",
//     {},
//     async () => {
//       await checkAuth();

//       const api = new ServerSideMarketplaceAPI();
//       await api.approveAgentSubmission(agentId, version, comment);
//       console.debug(`Approving agent ${agentId}`);
//       revalidatePath("/marketplace");
//     },
//   );
// }

// export async function rejectAgent(
//   agentId: string,
//   version: number,
//   comment: string,
// ) {
//   return await Sentry.withServerActionInstrumentation(
//     "rejectAgent",
//     {},
//     async () => {
//       await checkAuth();
//       const api = new ServerSideMarketplaceAPI();
//       await api.rejectAgentSubmission(agentId, version, comment);
//       console.debug(`Rejecting agent ${agentId}`);
//       revalidatePath("/marketplace");
//     },
//   );
// }

// export async function getReviewableAgents() {
//   return await Sentry.withServerActionInstrumentation(
//     "getReviewableAgents",
//     {},
//     async () => {
//       await checkAuth();
//       const api = new ServerSideMarketplaceAPI();
//       return api.getAgentSubmissions();
//     },
//   );
// }

// export async function getFeaturedAgents(
//   page: number = 1,
//   pageSize: number = 10,
// ) {
//   return await Sentry.withServerActionInstrumentation(
//     "getFeaturedAgents",
//     {},
//     async () => {
//       await checkAuth();
//       const api = new ServerSideMarketplaceAPI();
//       const featured = await api.getFeaturedAgents(page, pageSize);
//       console.debug(`Getting featured agents ${featured.items.length}`);
//       return featured;
//     },
//   );
// }

// export async function getFeaturedAgent(agentId: string) {
//   return await Sentry.withServerActionInstrumentation(
//     "getFeaturedAgent",
//     {},
//     async () => {
//       await checkAuth();
//       const api = new ServerSideMarketplaceAPI();
//       const featured = await api.getFeaturedAgent(agentId);
//       console.debug(`Getting featured agent ${featured.agentId}`);
//       return featured;
//     },
//   );
// }

// export async function addFeaturedAgent(
//   agentId: string,
//   categories: string[] = ["featured"],
// ) {
//   return await Sentry.withServerActionInstrumentation(
//     "addFeaturedAgent",
//     {},
//     async () => {
//       await checkAuth();
//       const api = new ServerSideMarketplaceAPI();
//       await api.addFeaturedAgent(agentId, categories);
//       console.debug(`Adding featured agent ${agentId}`);
//       revalidatePath("/marketplace");
//     },
//   );
// }

// export async function removeFeaturedAgent(
//   agentId: string,
//   categories: string[] = ["featured"],
// ) {
//   return await Sentry.withServerActionInstrumentation(
//     "removeFeaturedAgent",
//     {},
//     async () => {
//       await checkAuth();
//       const api = new ServerSideMarketplaceAPI();
//       await api.removeFeaturedAgent(agentId, categories);
//       console.debug(`Removing featured agent ${agentId}`);
//       revalidatePath("/marketplace");
//     },
//   );
// }

// export async function getCategories() {
//   return await Sentry.withServerActionInstrumentation(
//     "getCategories",
//     {},
//     async () => {
//       await checkAuth();
//       const api = new ServerSideMarketplaceAPI();
//       const categories = await api.getCategories();
//       console.debug(
//         `Getting categories ${categories.unique_categories.length}`,
//       );
//       return categories;
//     },
//   );
// }

// export async function getNotFeaturedAgents(
//   page: number = 1,
//   pageSize: number = 100,
// ) {
//   return await Sentry.withServerActionInstrumentation(
//     "getNotFeaturedAgents",
//     {},
//     async () => {
//       await checkAuth();
//       const api = new ServerSideMarketplaceAPI();
//       const agents = await api.getNotFeaturedAgents(page, pageSize);
//       console.debug(`Getting not featured agents ${agents.items.length}`);
//       return agents;
//     },
//   );
// }

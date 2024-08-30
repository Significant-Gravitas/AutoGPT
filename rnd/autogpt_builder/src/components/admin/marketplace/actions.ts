"use server";
import MarketplaceAPI from "@/lib/marketplace-api";

export async function approveAgent(
  agentId: string,
  version: number,
  comment: string,
) {
  const api = new MarketplaceAPI();
  await api.approveAgentSubmission(agentId, version, comment);
  console.debug(`Approving agent ${agentId}`);
}

export async function rejectAgent(
  agentId: string,
  version: number,
  comment: string,
) {
  const api = new MarketplaceAPI();
  await api.rejectAgentSubmission(agentId, version, comment);
  console.debug(`Rejecting agent ${agentId}`);
}

export async function getReviewableAgents() {
  const api = new MarketplaceAPI();
  return api.getAgentSubmissions();
}

export async function getFeaturedAgents(
  page: number = 1,
  pageSize: number = 10,
) {
  const api = new MarketplaceAPI();
  const featured = await api.getFeaturedAgents(page, pageSize);
  console.debug(`Getting featured agents ${featured.agents.length}`);
  return featured;
}


export async function removeFeaturedAgent(agentId: string) {
  const api = new MarketplaceAPI();
  // await api.removeFeaturedAgent(agentId);
  console.debug(`Removing featured agent ${agentId}`);
}
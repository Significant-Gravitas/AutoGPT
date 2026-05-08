import { getSeededTestUser } from "./accounts";

// E2E Test Credentials and Constants
export const TEST_CREDENTIALS = getSeededTestUser("primary");

export function getTestUserWithLibraryAgents() {
  return TEST_CREDENTIALS;
}

// Dummy constant to help developers identify agents that don't need input
export const DummyInput = "DummyInput";

// This will be used for testing agent submission for test123@example.com
export const TEST_AGENT_DATA = {
  name: "E2E Calculator Agent",
  description:
    "A deterministic marketplace agent built from Calculator and Agent Output blocks for frontend E2E coverage.",
  image_urls: [
    "https://picsum.photos/seed/e2e-marketplace-1/200/300",
    "https://picsum.photos/seed/e2e-marketplace-2/200/301",
    "https://picsum.photos/seed/e2e-marketplace-3/200/302",
  ],
  video_url: "https://www.youtube.com/watch?v=test123",
  sub_heading: "A deterministic calculator agent for PR E2E coverage",
  categories: ["test", "demo", "frontend"],
  changes_summary: "Initial deterministic calculator submission",
} as const;

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
  // Seed with no images so cards render their solid-color fallback. Avoids
  // the external picsum.photos dependency that intermittently 504'd in CI.
  image_urls: [] as string[],
  video_url: "https://www.youtube.com/watch?v=test123",
  sub_heading: "A deterministic calculator agent for PR E2E coverage",
  categories: ["test", "demo", "frontend"],
  changes_summary: "Initial deterministic calculator submission",
} as const;

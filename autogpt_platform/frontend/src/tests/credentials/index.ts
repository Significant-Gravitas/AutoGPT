// E2E Test Credentials and Constants
export const TEST_CREDENTIALS = {
  email: "test123@gmail.com",
  password: "testpassword123",
} as const;

export function getTestUserWithLibraryAgents() {
  return TEST_CREDENTIALS;
}

// Dummy constant to help developers identify agents that don't need input
export const DummyInput = "DummyInput";

// This will be used for testing agent submission for test123@gmail.com
export const TEST_AGENT_DATA = {
  name: "Test Agent Submission",
  description:
    "This is a test agent submission specifically created for frontend testing purposes.",
  image_urls: [
    "https://picsum.photos/200/300",
    "https://picsum.photos/200/301",
    "https://picsum.photos/200/302",
  ],
  video_url: "https://www.youtube.com/watch?v=test123",
  sub_heading: "A test agent for frontend testing",
  categories: ["test", "demo", "frontend"],
  changes_summary: "Initial test submission",
} as const;

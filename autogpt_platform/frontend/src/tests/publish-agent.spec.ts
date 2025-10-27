import test from "@playwright/test";
import { LoginPage } from "./pages/login.page";
import { TEST_CREDENTIALS } from "./credentials";
import { getSelectors } from "./utils/selectors";
import {
  hasUrl,
  isDisabled,
  isEnabled,
  isHidden,
  isVisible,
} from "./utils/assertion";

test("user can publish an agent through the complete flow", async ({
  page,
}) => {
  const { getId, getText, getButton } = getSelectors(page);

  const loginPage = new LoginPage(page);
  await page.goto("/login");
  await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
  await hasUrl(page, "/marketplace");

  await page.goto("/marketplace");
  await getButton("Become a creator").click();

  const publishAgentModal = getId("publish-agent-modal");
  await isVisible(publishAgentModal, 10000);

  await isVisible(
    publishAgentModal.getByText(
      "Select your project that you'd like to publish",
    ),
  );

  const agentToSelect = publishAgentModal.getByTestId("agent-card").first();
  await agentToSelect.click();

  const nextButton = publishAgentModal.getByRole("button", {
    name: "Next",
    exact: true,
  });

  await isEnabled(nextButton);
  await nextButton.click();

  // 2. Adding details of agent
  await isVisible(getText("Write a bit of details about your agent"));

  const agentName = "Test Agent Name";

  const agentTitle = publishAgentModal.getByLabel("Title");
  await agentTitle.fill(agentName);

  const agentSubheader = publishAgentModal.getByLabel("Subheader");
  await agentSubheader.fill("Test Agent Subheader");

  const agentSlug = publishAgentModal.getByLabel("Slug");
  await agentSlug.fill("test-agent-slug");

  const youtubeInput = publishAgentModal.getByLabel("Youtube video link");
  await youtubeInput.fill("https://www.youtube.com/watch?v=test");

  const categorySelect = publishAgentModal.locator(
    'select[aria-hidden="true"]',
  );
  await categorySelect.selectOption({ value: "other" });

  const descriptionInput = publishAgentModal.getByLabel("Description");
  await descriptionInput.fill(
    "This is a test agent description for the automated test.",
  );

  await isEnabled(publishAgentModal.getByRole("button", { name: "Submit" }));
});

test("should display appropriate content in agent creation modal when user is logged out", async ({
  page,
}) => {
  const { getText, getButton } = getSelectors(page);

  await page.goto("/marketplace");
  await getButton("Become a creator").click();

  await isVisible(
    getText(
      "Log in or create an account to publish your agents to the marketplace and join a community of creators",
    ),
  );
});

test("should validate all form fields in publish agent form", async ({
  page,
}) => {
  const { getId, getText, getButton } = getSelectors(page);

  const loginPage = new LoginPage(page);
  await page.goto("/login");
  await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
  await hasUrl(page, "/marketplace");

  await page.goto("/marketplace");
  await getButton("Become a creator").click();

  const publishAgentModal = getId("publish-agent-modal");
  await isVisible(publishAgentModal, 10000);

  const agentToSelect = publishAgentModal.getByTestId("agent-card").first();
  await agentToSelect.click();

  const nextButton = publishAgentModal.getByRole("button", {
    name: "Next",
    exact: true,
  });
  await nextButton.click();

  await isVisible(getText("Write a bit of details about your agent"));

  // Get form elements
  const agentTitle = publishAgentModal.getByLabel("Title");
  const agentSubheader = publishAgentModal.getByLabel("Subheader");
  const agentSlug = publishAgentModal.getByLabel("Slug");
  const youtubeInput = publishAgentModal.getByLabel("Youtube video link");
  const categorySelect = publishAgentModal.locator(
    'select[aria-hidden="true"]',
  );
  const descriptionInput = publishAgentModal.getByLabel("Description");
  const submitButton = publishAgentModal.getByRole("button", {
    name: "Submit",
  });

  async function clearForm() {
    await agentTitle.clear();
    await agentSubheader.clear();
    await agentSlug.clear();
    await youtubeInput.clear();
    await descriptionInput.clear();
  }

  // 1. Test required field validations
  await clearForm();
  await submitButton.click();

  await isVisible(publishAgentModal.getByText("Title is required"));
  await isVisible(publishAgentModal.getByText("Subheader is required"));
  await isVisible(publishAgentModal.getByText("Slug is required"));
  await isVisible(publishAgentModal.getByText("Category is required"));
  await isVisible(publishAgentModal.getByText("Description is required"));

  // 2. Test field length limits
  await clearForm();

  // Test title length limit (100 characters)
  const longTitle = "a".repeat(101);
  await agentTitle.fill(longTitle);
  await agentTitle.blur();
  await isVisible(
    publishAgentModal.getByText("Title must be less than 100 characters"),
  );

  // Test subheader length limit (200 characters)
  const longSubheader = "b".repeat(201);
  await agentSubheader.fill(longSubheader);
  await agentSubheader.blur();
  await isVisible(
    publishAgentModal.getByText("Subheader must be less than 200 characters"),
  );

  // Test slug length limit (50 characters)
  const longSlug = "c".repeat(51);
  await agentSlug.fill(longSlug);
  await agentSlug.blur();
  await isVisible(
    publishAgentModal.getByText("Slug must be less than 50 characters"),
  );

  // Test description length limit (1000 characters)
  const longDescription = "d".repeat(1001);
  await descriptionInput.fill(longDescription);
  await descriptionInput.blur();
  await isVisible(
    publishAgentModal.getByText(
      "Description must be less than 1000 characters",
    ),
  );

  // Test invalid characters in slug
  await agentSlug.fill("Invalid Slug With Spaces");
  await agentSlug.blur();
  await isVisible(
    publishAgentModal.getByText(
      "Slug can only contain lowercase letters, numbers, and hyphens",
    ),
  );

  await agentSlug.clear();
  await agentSlug.fill("InvalidSlugWithCapitals");
  await agentSlug.blur();
  await isVisible(
    publishAgentModal.getByText(
      "Slug can only contain lowercase letters, numbers, and hyphens",
    ),
  );

  await agentSlug.clear();
  await agentSlug.fill("invalid-slug-with-@#$");
  await agentSlug.blur();
  await isVisible(
    publishAgentModal.getByText(
      "Slug can only contain lowercase letters, numbers, and hyphens",
    ),
  );

  // Test valid slug format should not show error
  await agentSlug.clear();
  await agentSlug.fill("valid-slug-123");
  await agentSlug.blur();
  await page.waitForTimeout(500);

  await isHidden(
    publishAgentModal.getByText(
      "Slug can only contain lowercase letters, numbers, and hyphens",
    ),
  );

  // Test invalid YouTube URL
  await youtubeInput.fill("https://www.google.com/invalid-url");
  await youtubeInput.blur();
  await isVisible(
    publishAgentModal.getByText("Please enter a valid YouTube URL"),
  );

  await youtubeInput.clear();
  await youtubeInput.fill("not-a-url-at-all");
  await youtubeInput.blur();
  await isVisible(
    publishAgentModal.getByText("Please enter a valid YouTube URL"),
  );

  // Test valid YouTube URLs should not show error
  await youtubeInput.clear();
  await youtubeInput.fill("https://www.youtube.com/watch?v=test");
  await youtubeInput.blur();
  await page.waitForTimeout(500);

  await isHidden(
    publishAgentModal.getByText("Please enter a valid YouTube URL"),
  );

  await youtubeInput.clear();
  await youtubeInput.fill("https://youtu.be/test123");
  await youtubeInput.blur();
  await page.waitForTimeout(500);

  await isHidden(
    publishAgentModal.getByText("Please enter a valid YouTube URL"),
  );

  // 5. Test submit button enabled/disabled state
  await clearForm();

  // Submit button should be disabled when form is empty
  await page.waitForTimeout(1000);
  await isDisabled(submitButton);

  // Fill all required fields with valid data
  await agentTitle.fill("Valid Title");
  await agentSubheader.fill("Valid Subheader");
  await agentSlug.fill("valid-slug");
  await categorySelect.selectOption({ value: "other" });
  await descriptionInput.fill("Valid description text");

  // Submit button should now be enabled
  await isEnabled(submitButton);
});

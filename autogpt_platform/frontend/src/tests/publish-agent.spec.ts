import path from "path";
import test from "@playwright/test";
import { LoginPage } from "./pages/login.page";
import { TEST_CREDENTIALS } from "./credentials";
import { getSelectors } from "./utils/selectors";
import { hasUrl, isVisible } from "./utils/assertion";

test.beforeEach(async ({ page }) => {
  const loginPage = new LoginPage(page);
  await page.goto("/login");
  await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
  await hasUrl(page, "/marketplace");
});

test("user can publish an agent through the complete flow", async ({ page }) => {
  const {getId, getText} = getSelectors(page);
  
  await page.goto("/marketplace");
  await getId("become-a-creator-btn").click();
  
  await isVisible(getText("Select your project that you'd like to publish"));
  const agentToSelect = getId("agent-to-select").first();
  await agentToSelect.waitFor({ state: "visible" });
  await agentToSelect.click();
  await getId("next-button").click();
  
  // 2. Adding details of agent
  await isVisible(getText("Write a bit of details about your agent"));
  await getId("agent-title-input").fill("Test Agent Title");
  await getId("agent-subheader-input").fill("Test Agent Subheader");
  await getId("agent-slug-input").fill("test-agent-slug");

  const fileChooserPromise = page.waitForEvent('filechooser');
  await getId("add-image-btn").click();
  const fileChooser = await fileChooserPromise;
  await fileChooser.setFiles(path.resolve(__dirname, "assets/placeholder.png"));
  await page.waitForSelector('img[alt="Selected Thumbnail"]', { timeout: 10000 });


  await getId("agent-youtube-input").fill("https://www.youtube.com/watch?v=test");
  await getId("agent-category-select").selectOption({ index: 2 });
  await getId("agent-description-textarea").fill("This is a test agent description for the automated test.");
  await getId("agent-submit-button").click();
  
  // 3. Agent reviewing page
  await isVisible(getText("Agent is awaiting review"));
  await getId("view-progrss-btn").click();
  
  await hasUrl(page, "/profile/dashboard");
});
import test, { expect } from "@playwright/test";
import path from "path";
import { getTestUserWithLibraryAgents } from "./credentials";
import { LibraryPage } from "./pages/library.page";
import { LoginPage } from "./pages/login.page";
import { hasUrl } from "./utils/assertion";
import { getSelectors } from "./utils/selectors";

test.describe("Library", () => {
  let libraryPage: LibraryPage;

  test.beforeEach(async ({ page }) => {
    libraryPage = new LibraryPage(page);

    await page.goto("/login");
    const loginPage = new LoginPage(page);
    const richUser = getTestUserWithLibraryAgents();
    await loginPage.login(richUser.email, richUser.password);
    await hasUrl(page, "/marketplace");
  });

  test("library page loads successfully", async ({ page }) => {
    const { getId } = getSelectors(page);
    await page.goto("/library");

    await expect(getId("search-bar").first()).toBeVisible();
    await expect(getId("upload-agent-button").first()).toBeVisible();
    await expect(getId("sort-by-dropdown").first()).toBeVisible();
  });

  test("agents are visible and cards work correctly", async ({ page }) => {
    await page.goto("/library");

    const agents = await libraryPage.getAgents();
    expect(agents.length).toBeGreaterThan(0);

    const firstAgent = agents[0];
    expect(firstAgent).toBeTruthy();

    await libraryPage.clickAgent(firstAgent);
    await hasUrl(page, `/library/agents/${firstAgent.id}`);

    await libraryPage.navigateToLibrary();

    const updatedAgents = await libraryPage.getAgents();
    const agentWithBuilder = updatedAgents.find((agent) =>
      agent.openInBuilderUrl.includes("/build"),
    );

    if (agentWithBuilder) {
      const [newPage] = await Promise.all([
        page.context().waitForEvent("page"),
        libraryPage.clickOpenInBuilder(agentWithBuilder),
      ]);
      await newPage.waitForLoadState();
      test.expect(newPage.url()).toContain(`/build`);
      await newPage.close();
    }
  });

  test("pagination works correctly", async ({ page }, testInfo) => {
    test.setTimeout(testInfo.timeout * 3);
    await page.goto("/library");

    const PAGE_SIZE = 20;
    const paginationResult = await libraryPage.testPagination();

    if (paginationResult.initialCount >= PAGE_SIZE) {
      expect(paginationResult.finalCount).toBeGreaterThanOrEqual(
        paginationResult.initialCount,
      );
      expect(paginationResult.hasMore).toBeTruthy();
    }

    await libraryPage.isPaginationWorking();

    const allAgents = await libraryPage.getAgentsWithPagination();
    test.expect(allAgents.length).toBeGreaterThan(0);

    const displayedCount = await libraryPage.getAgentCount();
    test.expect(allAgents.length).toEqual(displayedCount);
  });

  test("searching works correctly", async ({ page }) => {
    await page.goto("/library");

    const allAgents = await libraryPage.getAgents();
    expect(allAgents.length).toBeGreaterThan(0);

    const initialAgentCount = await libraryPage.getAgentCount();
    expect(initialAgentCount).toBeGreaterThan(0);

    const firstAgent = allAgents[0];
    await libraryPage.searchAgents(firstAgent.name);
    await libraryPage.waitForAgentsToLoad();

    const searchResults = await libraryPage.getAgents();
    expect(searchResults.length).toBeGreaterThan(0);

    const foundAgent = searchResults.find(
      (agent) => agent.name === firstAgent.name,
    );
    expect(foundAgent).toBeTruthy();

    const searchValue = await libraryPage.getSearchValue();
    expect(searchValue).toBe(firstAgent.name);

    const partialSearchTerm = firstAgent.name.substring(0, 3);
    await libraryPage.searchAgents(partialSearchTerm);
    await libraryPage.waitForAgentsToLoad();

    const partialSearchResults = await libraryPage.getAgents();
    expect(partialSearchResults.length).toBeGreaterThan(0);

    const matchingAgents = partialSearchResults.filter((agent) =>
      agent.name.toLowerCase().includes(partialSearchTerm.toLowerCase()),
    );
    expect(matchingAgents.length).toBeGreaterThan(0);

    await libraryPage.searchAgents("nonexistentagentnamethatdoesnotexist");
    const noResults = await libraryPage.getAgentCount();
    expect(noResults).toBe(0);

    const hasNoAgentsMessage = await libraryPage.hasNoAgentsMessage();
    expect(hasNoAgentsMessage).toBeTruthy();

    await libraryPage.clearSearch();
    await libraryPage.waitForAgentsToLoad();

    const clearedSearchCount = await libraryPage.getAgentCount();
    test.expect(clearedSearchCount).toEqual(initialAgentCount);

    const clearedSearchValue = await libraryPage.getSearchValue();
    test.expect(clearedSearchValue).toBe("");
  });

  test("pagination while searching works correctly", async ({
    page,
  }, testInfo) => {
    test.setTimeout(testInfo.timeout * 3);
    await page.goto("/library");

    const allAgents = await libraryPage.getAgents();
    test.expect(allAgents.length).toBeGreaterThan(0);

    const searchTerm = "Agent";

    await libraryPage.searchAgents(searchTerm);
    await libraryPage.waitForAgentsToLoad();

    const initialSearchResults = await libraryPage.getAgents();
    expect(initialSearchResults.length).toBeGreaterThan(0);

    const matchingResults = initialSearchResults.filter((agent) =>
      agent.name.toLowerCase().includes(searchTerm.toLowerCase()),
    );
    expect(matchingResults.length).toEqual(initialSearchResults.length);

    const PAGE_SIZE = 20;
    const searchPaginationResult = await libraryPage.testPagination();

    if (searchPaginationResult.initialCount >= PAGE_SIZE) {
      expect(searchPaginationResult.finalCount).toBeGreaterThanOrEqual(
        searchPaginationResult.initialCount,
      );

      const allPaginatedResults = await libraryPage.getAgentsWithPagination();
      const matchingPaginatedResults = allPaginatedResults.filter((agent) =>
        agent.name.toLowerCase().includes(searchTerm.toLowerCase()),
      );
      expect(matchingPaginatedResults.length).toEqual(
        allPaginatedResults.length,
      );
    } else {
    }

    await libraryPage.scrollAndWaitForNewAgents();

    const finalSearchResults = await libraryPage.getAgents();
    const finalMatchingResults = finalSearchResults.filter((agent) =>
      agent.name.toLowerCase().includes(searchTerm.toLowerCase()),
    );
    expect(finalMatchingResults.length).toEqual(finalSearchResults.length);

    const preservedSearchValue = await libraryPage.getSearchValue();
    expect(preservedSearchValue).toBe(searchTerm);

    await libraryPage.clearSearch();
    await libraryPage.waitForAgentsToLoad();

    const clearedResults = await libraryPage.getAgents();
    expect(clearedResults.length).toBeGreaterThanOrEqual(
      initialSearchResults.length,
    );
  });

  test("uploading an agent works correctly", async ({ page }) => {
    await page.goto("/library");

    await libraryPage.openUploadDialog();

    expect(await libraryPage.isUploadDialogVisible()).toBeTruthy();
    expect(await libraryPage.isUploadButtonEnabled()).toBeFalsy();

    const testAgentName = "Test Upload Agent";
    const testAgentDescription = "This is a test agent uploaded via automation";
    await libraryPage.fillUploadForm(testAgentName, testAgentDescription);

    const fileInput = page.locator('input[type="file"]');
    const testAgentPath = path.resolve(
      __dirname,
      "assets",
      "testing_agent.json",
    );
    await fileInput.setInputFiles(testAgentPath);

    // Wait for file to be processed and upload button to be enabled
    const uploadButton = page.getByRole("button", { name: "Upload" });
    await uploadButton.waitFor({ state: "visible", timeout: 10000 });
    await expect(uploadButton).toBeEnabled({ timeout: 10000 });

    expect(await libraryPage.isUploadButtonEnabled()).toBeTruthy();

    await page.getByRole("button", { name: "Upload" }).click();

    await page.waitForURL("**/build**", { timeout: 10000 });
    expect(page.url()).toContain("/build");

    await page.goto("/library");

    await libraryPage.searchAgents(testAgentName);
    await libraryPage.waitForAgentsToLoad();

    const searchResults = await libraryPage.getAgents();
    test.expect(searchResults.length).toBeGreaterThan(0);

    const uploadedAgent = searchResults.find((agent) =>
      agent.name.includes(testAgentName),
    );
    test.expect(uploadedAgent).toBeTruthy();

    if (uploadedAgent) {
      test.expect(uploadedAgent.name).toContain(testAgentName);
      test.expect(uploadedAgent.seeRunsUrl).toBeTruthy();
      test.expect(uploadedAgent.openInBuilderUrl).toBeTruthy();
    }

    await libraryPage.clearSearch();
    await libraryPage.waitForAgentsToLoad();
  });
});

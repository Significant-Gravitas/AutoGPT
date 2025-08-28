import { LibraryPage } from "./pages/library.page";
import path from "path";
import test, { expect } from "@playwright/test";
import { TEST_CREDENTIALS } from "./credentials";
import { LoginPage } from "./pages/login.page";
import { getSelectors } from "./utils/selectors";
import { hasUrl } from "./utils/assertion";

test.describe("Library", () => {
  let libraryPage: LibraryPage;

  test.beforeEach(async ({ page }) => {
    libraryPage = new LibraryPage(page);

    await page.goto("/login");
    const loginPage = new LoginPage(page);
    await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
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
      await libraryPage.clickOpenInBuilder(agentWithBuilder);
      await page.waitForURL("**/build**");
      test.expect(page.url()).toContain(`/build`);
    }
  });

  test("pagination works correctly", async ({ page }) => {
    await page.goto("/library");

    const paginationResult = await libraryPage.testPagination();

    if (paginationResult.initialCount >= 10) {
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

  test("sorting works correctly", async ({ page }) => {
    await page.goto("/library");

    const initialAgents = await libraryPage.getAgents();
    expect(initialAgents.length).toBeGreaterThan(0);

    await libraryPage.selectSortOption(page, "Creation Date");
    await libraryPage.waitForAgentsToLoad();

    const creationDateSortOption = await libraryPage.getCurrentSortOption();
    expect(creationDateSortOption).toContain("Creation Date");

    const creationDateAgents = await libraryPage.getAgents();
    expect(creationDateAgents.length).toBeGreaterThan(0);

    await libraryPage.selectSortOption(page, "Last Modified");
    await libraryPage.waitForAgentsToLoad();

    const lastModifiedSortOption = await libraryPage.getCurrentSortOption();
    expect(lastModifiedSortOption).toContain("Last Modified");

    const lastModifiedAgents = await libraryPage.getAgents();
    expect(lastModifiedAgents.length).toBeGreaterThan(0);

    if (initialAgents.length > 1) {
      const initialFirstAgentId = initialAgents[0].id;
      const creationDateFirstAgentId = creationDateAgents[0].id;
      const lastModifiedFirstAgentId = lastModifiedAgents[0].id;

      expect(
        creationDateFirstAgentId !== initialFirstAgentId ||
          lastModifiedFirstAgentId !== initialFirstAgentId ||
          creationDateFirstAgentId !== lastModifiedFirstAgentId,
      ).toBeTruthy();
    }

    expect(creationDateAgents.length).toEqual(initialAgents.length);
    expect(lastModifiedAgents.length).toEqual(initialAgents.length);
  });

  test("searching works correctly", async ({ page }) => {
    await page.goto("/library");

    const allAgents = await libraryPage.getAgents();
    expect(allAgents.length).toBeGreaterThan(0);

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

    const clearedSearchResults = await libraryPage.getAgents();
    test.expect(clearedSearchResults.length).toEqual(allAgents.length);

    const clearedSearchValue = await libraryPage.getSearchValue();
    test.expect(clearedSearchValue).toBe("");
  });

  test("pagination while searching works correctly", async ({ page }) => {
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

    const searchPaginationResult = await libraryPage.testPagination();

    if (searchPaginationResult.initialCount >= 10) {
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

    await page.waitForTimeout(1000);

    expect(await libraryPage.isUploadButtonEnabled()).toBeTruthy();

    await page.getByRole("button", { name: "Upload Agent" }).click();

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
      test.expect(uploadedAgent.description).toContain(testAgentDescription);
      test.expect(uploadedAgent.seeRunsUrl).toBeTruthy();
      test.expect(uploadedAgent.openInBuilderUrl).toBeTruthy();

      await libraryPage.clickAgent(uploadedAgent);
      await page.waitForURL(`**/library/agents/${uploadedAgent.id}**`, {
        timeout: 10000,
      });

      await page.getByRole("button", { name: "Delete agent" }).click();
      await page.waitForTimeout(500);
      await page.getByRole("button", { name: "Delete" }).click();

      await page.waitForTimeout(1000);
      await libraryPage.navigateToLibrary();
      await libraryPage.waitForAgentsToLoad();

      await libraryPage.searchAgents(testAgentName);
      await libraryPage.waitForAgentsToLoad();
      const deletedSearchResults = await libraryPage.getAgentCount();
      expect(deletedSearchResults).toBe(0);
    }
    await libraryPage.clearSearch();
    await libraryPage.waitForAgentsToLoad();
  });
});

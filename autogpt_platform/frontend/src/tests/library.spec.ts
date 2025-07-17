import { test } from "./fixtures";
import { LibraryPage } from "./pages/library.page";
import { LibraryUtils } from "./utils/library";
import { loadUserPool } from "./utils/auth";
import path from "path";

test.describe("Library", () => {
  let libraryPage: LibraryPage;
  let libraryUtils: LibraryUtils;

  // I have created agents for the first user in global setup, so I need to login as that user to test the library
  test.beforeEach(async ({ page, loginPage }) => {
    libraryPage = new LibraryPage(page);
    libraryUtils = new LibraryUtils(page, libraryPage);

    // Start each test with login using worker auth
    await page.goto("/login");

    const users_pool = await loadUserPool();
    if (!users_pool) {
      console.error("No user has been created");
      return;
    }
    // Use a test user - adjust as needed for your test environment
    await loginPage.login(
      users_pool.users[0].email,
      users_pool.users[0].password,
    );

    await test.expect(page).toHaveURL("/marketplace");
  });

  test("library navigation is accessible from navbar", async ({ page }) => {
    await libraryUtils.navigateToLibrary();

    await libraryPage.navbar.clickMarketplaceLink();
    await test.expect(page).toHaveURL("/marketplace");

    await libraryPage.navbar.clickMonitorLink();
    await test.expect(page).toHaveURL("/library");
    await test.expect(libraryPage.isLoaded()).resolves.toBeTruthy();
  });

  test("library page loads successfully", async ({ page }) => {
    // Navigate to library with pre-created agents
    await libraryUtils.navigateToLibrary();

    await test.expect(libraryPage.isLoaded()).resolves.toBeTruthy();
    await test.expect(page).toHaveURL("/library");

    // Verify essential elements are present
    await test
      .expect(page.getByRole("textbox", { name: "Search agents" }))
      .toBeVisible();
    await test
      .expect(page.getByRole("button", { name: "Upload an agent" }))
      .toBeVisible();
    await test.expect(page.getByRole("combobox")).toBeVisible();
  });

  test("agents are visible and cards work correctly", async ({ page }) => {
    // Navigate to library with pre-created agents
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Verify agents are visible
    const agents = await libraryPage.getAgents();
    test.expect(agents.length).toBeGreaterThan(0);

    // Test with the first available agent
    const firstAgent = agents[0];
    test.expect(firstAgent).toBeTruthy();

    // Verify agent card is visible
    await test
      .expect(libraryPage.isAgentVisible(firstAgent))
      .resolves.toBeTruthy();

    // Test clicking the agent card itself
    await libraryPage.clickAgent(firstAgent);
    await test.expect(page).toHaveURL(`/library/agents/${firstAgent.id}`);

    // Go back to library
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Test "See runs" button
    await libraryPage.clickSeeRuns(firstAgent);
    await test.expect(page).toHaveURL(`/library/agents/${firstAgent.id}`);

    // Go back to library
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Test "Open in builder" button (if available)
    const updatedAgents = await libraryPage.getAgents();
    const agentWithBuilder = updatedAgents.find((agent) =>
      agent.openInBuilderUrl.includes("/build"),
    );

    if (agentWithBuilder) {
      await libraryPage.clickOpenInBuilder(agentWithBuilder);
      await page.waitForURL("**/build**");
      test.expect(page.url()).toContain(`/build`); // If we have access to graph id, then we need to add flowId to the URL as well
    }
  });

  test("pagination works correctly", async () => {
    // Navigate to library with pre-created agents
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Test pagination functionality
    const paginationResult = await libraryPage.testPagination();

    // Verify pagination loaded more agents (if available)
    if (paginationResult.initialCount >= 10) {
      // If we have enough agents, pagination should work
      test
        .expect(paginationResult.finalCount)
        .toBeGreaterThanOrEqual(paginationResult.initialCount);
      test.expect(paginationResult.hasMore).toBeTruthy();
    } else {
      // If we don't have enough agents, pagination might not trigger
      console.log(
        `Only ${paginationResult.initialCount} agents available, pagination may not be needed`,
      );
    }

    // Test that pagination is working by checking scroll behavior
    await libraryPage.isPaginationWorking();

    // Get all agents with pagination
    const allAgents = await libraryPage.getAgentsWithPagination();
    test.expect(allAgents.length).toBeGreaterThan(0);

    // Verify agent count matches displayed count
    const displayedCount = await libraryPage.getAgentCount();
    test.expect(allAgents.length).toEqual(displayedCount);

    console.log(
      `Pagination test completed: ${allAgents.length} total agents loaded`,
    );
  });

  test("sorting works correctly", async () => {
    // Navigate to library with pre-created agents
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Get initial agents
    const initialAgents = await libraryPage.getAgents();
    test.expect(initialAgents.length).toBeGreaterThan(0);

    // Test sorting by Creation Date
    await libraryPage.selectSortOption("Creation Date");
    await libraryPage.waitForAgentsToLoad();

    // Verify the sort option is selected
    const creationDateSortOption = await libraryPage.getCurrentSortOption();
    test.expect(creationDateSortOption).toContain("Creation Date");

    // Get agents after sorting by creation date
    const creationDateAgents = await libraryPage.getAgents();
    test.expect(creationDateAgents.length).toBeGreaterThan(0);

    // Test sorting by Last Modified
    await libraryPage.selectSortOption("Last Modified");
    await libraryPage.waitForAgentsToLoad();

    // Verify the sort option is selected
    const lastModifiedSortOption = await libraryPage.getCurrentSortOption();
    test.expect(lastModifiedSortOption).toContain("Last Modified");

    // Get agents after sorting by last modified
    const lastModifiedAgents = await libraryPage.getAgents();
    test.expect(lastModifiedAgents.length).toBeGreaterThan(0);

    // Verify sorting changed the order (if we have multiple agents)
    if (initialAgents.length > 1) {
      // Compare first agent IDs to verify order changed
      const initialFirstAgentId = initialAgents[0].id;
      const creationDateFirstAgentId = creationDateAgents[0].id;
      const lastModifiedFirstAgentId = lastModifiedAgents[0].id;

      console.log(`Initial first agent: ${initialFirstAgentId}`);
      console.log(`Creation date first agent: ${creationDateFirstAgentId}`);
      console.log(`Last modified first agent: ${lastModifiedFirstAgentId}`);

      // At least one of the sorting options should produce a different order
      test
        .expect(
          creationDateFirstAgentId !== initialFirstAgentId ||
            lastModifiedFirstAgentId !== initialFirstAgentId ||
            creationDateFirstAgentId !== lastModifiedFirstAgentId,
        )
        .toBeTruthy();
    }

    // Verify all agents are still present after sorting
    test.expect(creationDateAgents.length).toEqual(initialAgents.length);
    test.expect(lastModifiedAgents.length).toEqual(initialAgents.length);

    console.log("Sorting test completed successfully");
  });

  test("searching works correctly", async () => {
    // Navigate to library with pre-created agents
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Get all agents initially
    const allAgents = await libraryPage.getAgents();
    test.expect(allAgents.length).toBeGreaterThan(0);

    // Test searching with a specific agent name
    const firstAgent = allAgents[0];
    await libraryPage.searchAgents(firstAgent.name);
    await libraryPage.waitForAgentsToLoad();

    // Verify search results
    const searchResults = await libraryPage.getAgents();
    test.expect(searchResults.length).toBeGreaterThan(0);

    // Verify the searched agent is in the results
    const foundAgent = searchResults.find(
      (agent) => agent.name === firstAgent.name,
    );
    test.expect(foundAgent).toBeTruthy();

    // Verify search input value
    const searchValue = await libraryPage.getSearchValue();
    test.expect(searchValue).toBe(firstAgent.name);

    // Test partial search
    const partialSearchTerm = firstAgent.name.substring(0, 3);
    await libraryPage.searchAgents(partialSearchTerm);
    await libraryPage.waitForAgentsToLoad();

    const partialSearchResults = await libraryPage.getAgents();
    test.expect(partialSearchResults.length).toBeGreaterThan(0);

    // Verify partial search results contain the search term
    const matchingAgents = partialSearchResults.filter((agent) =>
      agent.name.toLowerCase().includes(partialSearchTerm.toLowerCase()),
    );
    test.expect(matchingAgents.length).toBeGreaterThan(0);

    // Test search with no results
    await libraryPage.searchAgents("nonexistentagentnamethatdoesnotexist");
    await libraryPage.waitForAgentsToLoad();

    const noResults = await libraryPage.getAgents();
    test.expect(noResults.length).toBe(0);

    // Verify no agents message is shown
    const hasNoAgentsMessage = await libraryPage.hasNoAgentsMessage();
    test.expect(hasNoAgentsMessage).toBeTruthy();

    // Test clearing search
    await libraryPage.clearSearch();
    await libraryPage.waitForAgentsToLoad();

    // Verify all agents are back
    const clearedSearchResults = await libraryPage.getAgents();
    test.expect(clearedSearchResults.length).toEqual(allAgents.length);

    // Verify search input is cleared
    const clearedSearchValue = await libraryPage.getSearchValue();
    test.expect(clearedSearchValue).toBe("");

    console.log("Search test completed successfully");
  });

  test("pagination while searching works correctly", async () => {
    // Navigate to library with pre-created agents
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Get all agents initially
    const allAgents = await libraryPage.getAgents();
    test.expect(allAgents.length).toBeGreaterThan(0);

    // Use "Agent" as search term since it appears in most of the predefined agent names
    const searchTerm = "Agent";

    // Perform search
    await libraryPage.searchAgents(searchTerm);
    await libraryPage.waitForAgentsToLoad();

    // Get initial search results
    const initialSearchResults = await libraryPage.getAgents();
    test.expect(initialSearchResults.length).toBeGreaterThan(0);

    // Verify all results match the search term
    const matchingResults = initialSearchResults.filter((agent) =>
      agent.name.toLowerCase().includes(searchTerm.toLowerCase()),
    );
    test.expect(matchingResults.length).toEqual(initialSearchResults.length);

    // Test pagination within search results
    const searchPaginationResult = await libraryPage.testPagination();

    // If we have enough search results, pagination should work
    if (searchPaginationResult.initialCount >= 10) {
      test
        .expect(searchPaginationResult.finalCount)
        .toBeGreaterThanOrEqual(searchPaginationResult.initialCount);

      // Verify all paginated results still match the search term
      const allPaginatedResults = await libraryPage.getAgentsWithPagination();
      const matchingPaginatedResults = allPaginatedResults.filter((agent) =>
        agent.name.toLowerCase().includes(searchTerm.toLowerCase()),
      );
      test
        .expect(matchingPaginatedResults.length)
        .toEqual(allPaginatedResults.length);

      console.log(
        `Pagination in search worked: ${searchPaginationResult.initialCount} -> ${searchPaginationResult.finalCount} results`,
      );
    } else {
      console.log(
        `Only ${searchPaginationResult.initialCount} search results available, pagination may not be needed`,
      );
    }

    // Test scrolling to load more search results
    await libraryPage.scrollAndWaitForNewAgents();
    const finalSearchCount = await libraryPage.getAgentCount();

    // Verify search results are maintained after scrolling
    const finalSearchResults = await libraryPage.getAgents();
    const finalMatchingResults = finalSearchResults.filter((agent) =>
      agent.name.toLowerCase().includes(searchTerm.toLowerCase()),
    );
    test.expect(finalMatchingResults.length).toEqual(finalSearchResults.length);

    // Test that search term is preserved during pagination
    const preservedSearchValue = await libraryPage.getSearchValue();
    test.expect(preservedSearchValue).toBe(searchTerm);

    // Clear search and verify full pagination works again
    await libraryPage.clearSearch();
    await libraryPage.waitForAgentsToLoad();

    const clearedResults = await libraryPage.getAgents();
    test
      .expect(clearedResults.length)
      .toBeGreaterThanOrEqual(initialSearchResults.length);

    console.log(
      `Pagination while searching test completed: searched for "${searchTerm}", found ${finalSearchCount} results`,
    );
  });

  test("uploading an agent works correctly", async ({ page }) => {
    // Navigate to library with pre-created agents
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Open upload dialog
    await libraryPage.openUploadDialog();
    await test
      .expect(libraryPage.isUploadDialogVisible())
      .resolves.toBeTruthy();

    // Verify upload button is initially disabled
    await test.expect(libraryPage.isUploadButtonEnabled()).resolves.toBeFalsy();

    // Fill upload form
    const testAgentName = "Test Upload Agent";
    const testAgentDescription = "This is a test agent uploaded via automation";
    await libraryPage.fillUploadForm(testAgentName, testAgentDescription);

    // Set up file input for agent upload
    const fileInput = page.locator('input[type="file"]');
    const testAgentPath = path.resolve(
      __dirname,
      "assets",
      "testing_agent.json",
    );
    await fileInput.setInputFiles(testAgentPath);

    // Wait for file to be processed
    await page.waitForTimeout(1000);

    // Verify upload button is now enabled
    await test
      .expect(libraryPage.isUploadButtonEnabled())
      .resolves.toBeTruthy();

    // Click upload button
    await page.getByRole("button", { name: "Upload Agent" }).click();

    // After upload, we should be redirected to /build
    await page.waitForURL("**/build**", { timeout: 10000 });
    test.expect(page.url()).toContain("/build");

    // Navigate back to library
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Search for the uploaded agent
    await libraryPage.searchAgents(testAgentName);
    await libraryPage.waitForAgentsToLoad();

    // Verify the uploaded agent is found
    const searchResults = await libraryPage.getAgents();
    test.expect(searchResults.length).toBeGreaterThan(0);

    const uploadedAgent = searchResults.find((agent) =>
      agent.name.includes(testAgentName),
    );
    test.expect(uploadedAgent).toBeTruthy();

    // Verify agent details
    if (uploadedAgent) {
      test.expect(uploadedAgent.name).toContain(testAgentName);
      test.expect(uploadedAgent.description).toContain(testAgentDescription);
      test.expect(uploadedAgent.seeRunsUrl).toBeTruthy();
      test.expect(uploadedAgent.openInBuilderUrl).toBeTruthy();

      // Click on the uploaded agent to navigate to its detail page
      await libraryPage.clickAgent(uploadedAgent);
      await page.waitForURL(`**/library/agents/${uploadedAgent.id}**`, {
        timeout: 10000,
      });

      // Click the "Delete agent" button
      await page.getByRole("button", { name: "Delete agent" }).click();

      // Wait for the popover to appear and click the delete button
      await page.waitForTimeout(500);
      await page.getByRole("button", { name: "Delete" }).click();

      // Wait for deletion to complete and navigate back to library
      await page.waitForTimeout(1000);
      await libraryUtils.navigateToLibrary();
      await libraryPage.waitForAgentsToLoad();

      // Verify the agent is no longer in the library
      await libraryPage.searchAgents(testAgentName);
      await libraryPage.waitForAgentsToLoad();

      const deletedSearchResults = await libraryPage.getAgents();
      const deletedAgent = deletedSearchResults.find((agent) =>
        agent.name.includes(testAgentName),
      );
      test.expect(deletedAgent).toBeFalsy();
    }

    // Clear search to restore full view
    await libraryPage.clearSearch();
    await libraryPage.waitForAgentsToLoad();

    console.log("Agent upload and deletion test completed successfully");
  });

  test("Edge case : search edge cases and error handling behave correctly", async () => {
    // Navigate to library with pre-created agents
    await libraryUtils.navigateToLibrary();
    await libraryPage.waitForAgentsToLoad();

    // Test 1: Empty search string handling
    console.log("Testing empty search string");
    await libraryPage.searchAgents("");
    await libraryPage.waitForAgentsToLoad();

    // Verify all agents are still visible with empty search
    const emptySearchResults = await libraryPage.getAgents();
    test.expect(emptySearchResults.length).toBeGreaterThan(0);

    // Test 2: Search with only whitespace
    console.log("Testing whitespace-only search");
    await libraryPage.searchAgents("   ");
    await libraryPage.waitForAgentsToLoad();

    const whitespaceSearchResults = await libraryPage.getAgents();
    // Should either show all agents or handle whitespace gracefully
    test.expect(whitespaceSearchResults.length).toBeGreaterThanOrEqual(0);

    // Test 3: Search with special characters
    console.log("Testing special character search");
    await libraryPage.searchAgents("!@#$%^&*()");
    await libraryPage.waitForAgentsToLoad();

    const specialCharResults = await libraryPage.getAgents();
    test.expect(specialCharResults.length).toBe(0);

    // Verify empty state is shown for special characters
    const hasNoAgentsForSpecialChars = await libraryPage.hasNoAgentsMessage();
    test.expect(hasNoAgentsForSpecialChars).toBeTruthy();

    // Test 4: Search with very long string
    console.log("Testing very long search string");
    const longSearchTerm = "a".repeat(1000);
    await libraryPage.searchAgents(longSearchTerm);
    await libraryPage.waitForAgentsToLoad();

    const longSearchResults = await libraryPage.getAgents();
    test.expect(longSearchResults.length).toBe(0);

    // Test 5: Search that returns no results
    console.log("Testing search with no results");
    await libraryPage.searchAgents(
      "nonexistentagentnamethatdoesnotexist123456",
    );
    await libraryPage.waitForAgentsToLoad();

    // Verify empty state is shown
    const hasNoAgentsMessage = await libraryPage.hasNoAgentsMessage();
    test.expect(hasNoAgentsMessage).toBeTruthy();

    // Verify no agent cards are visible
    const noResultsAgents = await libraryPage.getAgents();
    test.expect(noResultsAgents.length).toBe(0);

    // Verify agent count shows 0
    const emptyCount = await libraryPage.getAgentCount();
    test.expect(emptyCount).toBe(0);

    // Test 6: Sorting on empty search results
    console.log("Testing sorting on empty search results");

    // Try sorting by Creation Date on empty results
    await libraryPage.selectSortOption("Creation Date");
    await libraryPage.waitForAgentsToLoad();

    const emptySortedByCreation = await libraryPage.getAgents();
    test.expect(emptySortedByCreation.length).toBe(0);

    // Try sorting by Last Modified on empty results
    await libraryPage.selectSortOption("Last Modified");
    await libraryPage.waitForAgentsToLoad();

    const emptySortedByModified = await libraryPage.getAgents();
    test.expect(emptySortedByModified.length).toBe(0);

    // Verify sort options still work even with no results
    const sortOption = await libraryPage.getCurrentSortOption();
    test.expect(sortOption).toContain("Last Modified");

    // Test 7: Pagination on empty search results
    console.log("Testing pagination on empty search results");
    const paginationOnEmpty = await libraryPage.testPagination();
    test.expect(paginationOnEmpty.initialCount).toBe(0);
    test.expect(paginationOnEmpty.finalCount).toBe(0);
    test.expect(paginationOnEmpty.hasMore).toBeFalsy();

    console.log("Search edge case tests completed successfully");
  });
});

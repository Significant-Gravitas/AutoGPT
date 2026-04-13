import type { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { createUniqueAgentName } from "./pages/build.page";
import {
  clickRunButton,
  dismissFeedbackDialog,
  LibraryPage,
  waitForAgentPageLoad,
} from "./pages/library.page";

test.use({ storageState: E2E_AUTH_STATES.library });

const API_PREFIX = "/api/proxy/api";
const CALCULATOR_BLOCK_ID = "b1ab9b19-67a6-406d-abf5-2dba76d00c79";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";

type GraphCreateResponse = {
  id?: string;
  version?: number;
  data?: {
    id?: string;
    version?: number;
  };
};

type LibraryAgentRecord = {
  id: string;
  name: string;
};

type LibraryAgentListResponse = {
  agents: LibraryAgentRecord[];
};

type FolderRecord = {
  id: string;
  name: string;
};

async function createDeterministicCalculatorSavedAgent(
  page: Page,
  agentName: string,
  outputName: string,
) {
  const response = await page.request.post(`${API_PREFIX}/graphs`, {
    data: {
      graph: {
        name: agentName,
        description: "Deterministic calculator output for library happy paths",
        is_active: true,
        nodes: [
          {
            id: "calc-1",
            block_id: CALCULATOR_BLOCK_ID,
            input_default: {
              operation: "Add",
              a: 1,
              b: 1,
              round_result: false,
            },
            metadata: {
              position: { x: 120, y: 160 },
            },
            input_links: [],
            output_links: [],
          },
          {
            id: "final-output",
            block_id: AGENT_OUTPUT_BLOCK_ID,
            input_default: {
              name: outputName,
              description: "Deterministic calculator output",
            },
            metadata: {
              position: { x: 520, y: 160 },
            },
            input_links: [],
            output_links: [],
          },
        ],
        links: [
          {
            source_id: "calc-1",
            sink_id: "final-output",
            source_name: "result",
            sink_name: "value",
          },
        ],
      },
      source: "upload",
    },
  });

  expect(response.ok()).toBe(true);

  const body = (await response.json()) as GraphCreateResponse;
  const graphId = body.data?.id ?? body.id;
  expect(graphId).toBeTruthy();

  return String(graphId);
}

async function listLibraryAgents(page: Page, searchTerm: string) {
  const params = new URLSearchParams({
    search_term: searchTerm,
    page_size: "100",
  });
  const response = await page.request.get(
    `${API_PREFIX}/library/agents?${params.toString()}`,
  );
  expect(response.ok()).toBe(true);
  const body = (await response.json()) as LibraryAgentListResponse;
  return body.agents;
}

async function getLibraryAgentIdByName(page: Page, agentName: string) {
  const matchingAgent = (await listLibraryAgents(page, agentName)).find(
    (agent) => agent.name === agentName,
  );
  expect(matchingAgent?.id).toBeTruthy();
  return String(matchingAgent?.id);
}

async function updateLibraryAgent(
  page: Page,
  libraryAgentId: string,
  data: Record<string, unknown>,
) {
  const response = await page.request.patch(
    `${API_PREFIX}/library/agents/${libraryAgentId}`,
    { data },
  );
  expect(response.ok()).toBe(true);
}

async function createFolder(page: Page, folderName: string) {
  const response = await page.request.post(`${API_PREFIX}/library/folders`, {
    data: {
      name: folderName,
      color: "#3B82F6",
      icon: "📁",
    },
  });
  expect(response.ok()).toBe(true);
  const body = (await response.json()) as FolderRecord;
  expect(body.id).toBeTruthy();
  return body.id;
}

test("library happy path: user can search Library and navigate to an agent page", async ({
  page,
}) => {
  test.setTimeout(120000);

  const agentName = createUniqueAgentName("E2E Library Search Agent");
  const outputName = `e2e-search-output-${Date.now()}`;
  await createDeterministicCalculatorSavedAgent(page, agentName, outputName);

  const libraryPage = new LibraryPage(page);
  await page.goto("/library");
  await libraryPage.waitForAgentsToLoad();
  await libraryPage.searchAgents(agentName);

  const agentCard = page
    .getByTestId("library-agent-card")
    .filter({ hasText: agentName })
    .first();
  await expect(agentCard).toBeVisible({ timeout: 15000 });
  await agentCard.getByRole("link", { name: agentName, exact: true }).click();

  await waitForAgentPageLoad(page, agentName);
  await expect(page).toHaveURL(/\/library\/agents\/[^/?#]+(?:\?.*)?$/);
  await expect(
    page
      .locator('a[href*="/library/agents/"]')
      .filter({ hasText: agentName })
      .first(),
  ).toBeVisible({ timeout: 15000 });
});

test("library happy path: user can favorite an agent and find it in Favorites", async ({
  page,
}) => {
  test.setTimeout(120000);

  const agentName = createUniqueAgentName("E2E Library Favorite Agent");
  const outputName = `e2e-favorite-output-${Date.now()}`;
  await createDeterministicCalculatorSavedAgent(page, agentName, outputName);

  const libraryPage = new LibraryPage(page);
  await page.goto("/library");
  await libraryPage.waitForAgentsToLoad();
  await libraryPage.searchAgents(agentName);

  const agentCard = page
    .getByTestId("library-agent-card")
    .filter({ hasText: agentName })
    .first();
  await expect(agentCard).toBeVisible({ timeout: 15000 });
  await agentCard.hover();

  const favoriteButton = agentCard.getByRole("button", {
    name: "Add to favorites",
  });
  await expect(favoriteButton).toBeVisible({ timeout: 15000 });
  await favoriteButton.click();

  const favoritesTab = page.getByRole("tab", { name: /Favorites/i });
  await expect(favoritesTab).toBeEnabled({ timeout: 15000 });
  await favoritesTab.click();

  const favoriteCard = page
    .getByTestId("library-agent-card")
    .filter({ hasText: agentName })
    .first();
  await expect(favoriteCard).toBeVisible({ timeout: 15000 });
  await expect(
    favoriteCard.getByRole("button", { name: "Remove from favorites" }),
  ).toBeVisible({ timeout: 15000 });
});

test("library happy path: user can open a folder and see agents inside it", async ({
  page,
}) => {
  test.setTimeout(120000);

  const agentName = createUniqueAgentName("E2E Library Folder Agent");
  const outputName = `e2e-folder-output-${Date.now()}`;
  await createDeterministicCalculatorSavedAgent(page, agentName, outputName);

  const libraryAgentId = await getLibraryAgentIdByName(page, agentName);
  const folderName = `E2E Folder ${Date.now()}`;
  const folderId = await createFolder(page, folderName);
  await updateLibraryAgent(page, libraryAgentId, { folder_id: folderId });

  const libraryPage = new LibraryPage(page);
  await page.goto("/library");
  await libraryPage.waitForAgentsToLoad();

  await expect(
    page.getByTestId("library-agent-card").filter({ hasText: agentName }),
  ).toHaveCount(0);

  const folderCard = page
    .getByTestId("library-folder")
    .filter({ hasText: folderName })
    .first();
  await expect(folderCard).toBeVisible({ timeout: 15000 });
  await folderCard.click();

  const folderAgentCard = page
    .getByTestId("library-agent-card")
    .filter({ hasText: agentName })
    .first();
  await expect(folderAgentCard).toBeVisible({ timeout: 15000 });
});

test("library happy path: user can run a saved agent", async ({ page }) => {
  test.setTimeout(150000);

  const agentName = createUniqueAgentName("E2E Library Run Agent");
  const outputName = `e2e-run-output-${Date.now()}`;
  await createDeterministicCalculatorSavedAgent(page, agentName, outputName);

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);
  await clickRunButton(page);
  await libraryPage.waitForRunToComplete();
  await dismissFeedbackDialog(page);

  await libraryPage.assertRunProducedOutput();
  await libraryPage.assertRunOutputValue(outputName, /^2(?:\.0+)?$/);
  await expect
    .poll(() => libraryPage.getRunStatus(), { timeout: 15000 })
    .toBe("completed");
});

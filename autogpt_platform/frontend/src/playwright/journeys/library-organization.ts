import { randomBytes } from "crypto";
import {
  expect,
  type Locator,
  type Page,
  type Response,
  type TestInfo,
} from "@playwright/test";
import { authHeaders, signIn, type JourneyUser } from "./journey-user";

// The Library organization journey's steps: workflows made through the API,
// as other journeys make them, then folders, favorites, search and sort in
// the Library page, the way a person uses them.

const AGENT_INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";
const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;

// The colours the folder dialogs offer (components/folder-constants.ts).
export const FOLDER_COLOR = {
  Blue: "#3B82F6",
  Purple: "#A855F7",
  Green: "#10B981",
  Orange: "#F97316",
} as const;
type FolderColorLabel = keyof typeof FOLDER_COLOR;

// The sort menu's options and the `sort` each one puts in the URL.
export const SORTS = {
  "Creation Date": "createdAt",
  "Last Modified": "updatedAt",
  "Last Run": "lastRunAt",
} as const;
export type SortLabel = keyof typeof SORTS;

export interface Workflow {
  name: string;
  graphID: string;
  graphVersion: number;
  libraryAgentID: string;
}

export interface Folder {
  id: string;
  name: string;
  icon: string | null;
  color: string | null;
  parent_id: string | null;
  agent_count: number;
  subfolder_count: number;
}

export interface LibraryEntry {
  id: string;
  folder_id: string | null;
  is_favorite: boolean;
  created_at: string;
  updated_at: string;
  last_run_at: string | null;
}

/** A word no other journey's names contain, so a search finds only ours. */
export function uniqueStamp() {
  return `${Date.now()}${randomBytes(2).toString("hex")}`;
}

// ── Set-up through the API ─────────────────────────────────────────────

/** Makes one workflow per label, in order: the order they were made in is
 * the Creation Date sort's. */
export async function createWorkflows(
  page: Page,
  stamp: string,
  labels: string[],
) {
  const workflows: Workflow[] = [];
  for (const label of labels) {
    workflows.push(
      await createWorkflow(page, `Library journey ${stamp} ${label}`),
    );
  }
  return workflows;
}

async function createWorkflow(page: Page, name: string): Promise<Workflow> {
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: { source: "upload", graph: echoGraph(name) },
  });
  expect(response.ok(), `creating ${name}: ${response.status()}`).toBe(true);
  const graph = (await response.json()) as { id: string; version: number };
  const entry = await page.request.get(
    `/api/proxy/api/library/agents/by-graph/${graph.id}`,
  );
  expect(entry.ok(), `reading ${name}'s Library entry`).toBe(true);
  return {
    name,
    graphID: graph.id,
    graphVersion: graph.version,
    libraryAgentID: ((await entry.json()) as { id: string }).id,
  };
}

// AgentInput "note" -> AgentOutput "reply".
function echoGraph(name: string) {
  return {
    name,
    description: "Made by the Library organization journey",
    is_active: true,
    nodes: [
      graphNode("note", AGENT_INPUT_BLOCK_ID, { name: "note" }, 0),
      graphNode("reply", AGENT_OUTPUT_BLOCK_ID, { name: "reply" }, 400),
    ],
    links: [
      {
        source_id: "note",
        source_name: "result",
        sink_id: "reply",
        sink_name: "value",
      },
    ],
  };
}

function graphNode(
  id: string,
  blockID: string,
  input: Record<string, unknown>,
  x: number,
) {
  return {
    id,
    block_id: blockID,
    input_default: input,
    metadata: { position: { x, y: 0 } },
    input_links: [],
    output_links: [],
  };
}

/** Makes a folder the way Otto does, which is also how a subfolder is made:
 * the Library page has no way to nest one. */
export async function createFolder(
  page: Page,
  folder: {
    name: string;
    color: FolderColorLabel;
    icon: string;
    parentID?: string;
  },
) {
  const response = await page.request.post("/api/proxy/api/library/folders", {
    data: {
      name: folder.name,
      color: FOLDER_COLOR[folder.color],
      icon: folder.icon,
      parent_id: folder.parentID ?? null,
    },
  });
  expect(
    response.status(),
    `creating the folder ${folder.name}: ${await response.text()}`,
  ).toBe(201);
  return (await response.json()) as Folder;
}

/** Moves each workflow in its own call, so each move has its own time. */
export async function moveWorkflows(
  page: Page,
  workflows: Workflow[],
  folderID: string | null,
) {
  for (const workflow of workflows) {
    const response = await page.request.post(
      "/api/proxy/api/library/folders/agents/bulk-move",
      { data: { agent_ids: [workflow.libraryAgentID], folder_id: folderID } },
    );
    expect(response.ok(), `moving ${workflow.name}: ${response.status()}`).toBe(
      true,
    );
  }
}

/** Runs the workflow to its end: a run is what the Last Run sort orders by. */
export async function runToEnd(page: Page, workflow: Workflow) {
  const started = await page.request.post(
    `/api/proxy/api/graphs/${workflow.graphID}/execute/${workflow.graphVersion}`,
    { data: { inputs: { note: workflow.name }, credentials_inputs: {} } },
  );
  expect(started.ok(), `running ${workflow.name}: ${started.status()}`).toBe(
    true,
  );
  const { id } = (await started.json()) as { id: string };
  await expect
    .poll(
      async () => {
        const run = await page.request.get(
          `/api/proxy/api/graphs/${workflow.graphID}/executions/${id}`,
        );
        return ((await run.json()) as { status: string }).status;
      },
      { message: `${workflow.name}'s run should end`, timeout: 30_000 },
    )
    .toMatch(RUN_ENDED);
}

export async function readEntry(page: Page, workflow: Workflow) {
  const response = await page.request.get(
    `/api/proxy/api/library/agents/${workflow.libraryAgentID}`,
  );
  expect(response.ok(), `reading ${workflow.name}: ${response.status()}`).toBe(
    true,
  );
  return (await response.json()) as LibraryEntry;
}

export async function readFolders(page: Page, parentID?: string) {
  const response = await page.request.get("/api/proxy/api/library/folders", {
    params: parentID ? { parent_id: parentID } : {},
  });
  expect(response.ok(), `listing folders: ${response.status()}`).toBe(true);
  return ((await response.json()) as { folders: Folder[] }).folders;
}

/** The folder's status code and, when it is found, the folder. */
export async function readFolder(page: Page, folderID: string) {
  const response = await page.request.get(
    `/api/proxy/api/library/folders/${folderID}`,
  );
  return {
    status: response.status(),
    folder: response.ok() ? ((await response.json()) as Folder) : null,
  };
}

/** The journey's workflows in the order the given sort should list them,
 * worked out from each workflow's own timestamps. */
export async function orderBy(
  page: Page,
  workflows: Workflow[],
  sort: SortLabel,
) {
  const rows = await Promise.all(
    workflows.map(async (workflow) => ({
      name: workflow.name,
      entry: await readEntry(page, workflow),
    })),
  );
  const time = (value: string | null) =>
    value ? new Date(value).getTime() : null;
  rows.sort((a, b) => {
    const created =
      time(a.entry.created_at)! - time(b.entry.created_at)! ||
      a.entry.id.localeCompare(b.entry.id);
    // The API's updated_at is the later of the graph's and the entry's
    // times; the list sorts by the entry's alone. They agree while the
    // entry's is the later one, as it is for workflows made like these.
    const updated = time(b.entry.updated_at)! - time(a.entry.updated_at)!;
    if (sort === "Creation Date") return created;
    if (sort === "Last Modified") return updated;
    const [runA, runB] = [time(a.entry.last_run_at), time(b.entry.last_run_at)];
    if (runA === runB) return updated;
    if (runA === null) return 1;
    if (runB === null) return -1;
    return runB - runA;
  });
  return rows.map((row) => row.name);
}

// ── The Library page ───────────────────────────────────────────────────

export async function openLibrary(page: Page) {
  await page.goto("/library");
  // The page writes its sort into the URL with a navigation as it loads;
  // act once that has happened.
  await expect(page).toHaveURL(/[?&]sort=/);
  await expect(allTab(page)).toBeVisible();
}

export function allTab(page: Page) {
  return page.getByRole("tab", { name: /^All\b/ });
}

export function favoritesTab(page: Page) {
  return page.getByRole("tab", { name: /^Favorites\b/ });
}

export function workflowCard(page: Page, workflow: Workflow) {
  return page.locator(
    `[data-testid="library-agent-card"][data-agent-id="${workflow.libraryAgentID}"]`,
  );
}

export function folderCard(page: Page, folderID: string) {
  return page.locator(
    `[data-testid="library-folder"][data-folder-id="${folderID}"]`,
  );
}

/** Which of the journey's workflows the page lists, top to bottom. A preview
 * account may have other workflows, which this leaves out, and more than a
 * page of them: each read first scrolls to the last card, as a person would,
 * so a poll keeps loading pages until the journey's workflows are there. */
export async function listedWorkflows(page: Page, workflows: Workflow[]) {
  const cards = page.getByTestId("library-agent-card");
  if ((await cards.count()) > 0) {
    await cards
      .last()
      .scrollIntoViewIfNeeded({ timeout: 2_000 })
      .catch(() => {});
  }
  const ids = await cards.evaluateAll((nodes) =>
    nodes.map((card) => card.getAttribute("data-agent-id")),
  );
  return ids.flatMap((id) =>
    workflows
      .filter((workflow) => workflow.libraryAgentID === id)
      .map((workflow) => workflow.name),
  );
}

export async function expectListed(
  page: Page,
  workflows: Workflow[],
  expected: Workflow[],
  message: string,
) {
  await expect
    .poll(async () => (await listedWorkflows(page, workflows)).sort(), {
      message,
    })
    .toEqual(expected.map((workflow) => workflow.name).sort());
}

/** Create folder, filled in as a person does: a name, a colour and the
 * picker's first emoji. */
export async function createFolderInDialog(
  page: Page,
  folder: { name: string; color: FolderColorLabel },
) {
  await page.getByRole("button", { name: "Create folder" }).click();
  const dialog = page.getByRole("dialog", { name: "Create Folder" });
  await dialog.getByRole("textbox", { name: "Folder name" }).fill(folder.name);
  await pickColor(page, dialog, folder.color);
  const icon = await pickIcon(dialog, 0);
  const created = page.waitForResponse(isCall("POST", "/api/library/folders"));
  await dialog.getByRole("button", { name: "Create", exact: true }).click();
  const response = await created;
  expect(response.status(), "creating the folder").toBe(201);
  expect(
    response.request().postDataJSON(),
    "the folder should be made as it was filled in",
  ).toEqual({ name: folder.name, color: FOLDER_COLOR[folder.color], icon });
  await expect(dialog).toBeHidden();
  return (await response.json()) as Folder;
}

/** Edit folder: a new name, a new colour and the picker's second emoji. */
export async function editFolderInDialog(
  page: Page,
  folder: Folder,
  edit: { name: string; color: FolderColorLabel },
) {
  await folderCard(page, folder.id)
    .getByRole("button", { name: "Edit folder" })
    .click();
  const dialog = page.getByRole("dialog", { name: "Edit Folder" });
  const name = dialog.getByRole("textbox", { name: "Folder name" });
  await expect(name, "the dialog should open on the folder").toHaveValue(
    folder.name,
  );
  await name.fill(edit.name);
  await pickColor(page, dialog, edit.color);
  const icon = await pickIcon(dialog, 1);
  const saved = page.waitForResponse(
    isCall("PATCH", `/api/library/folders/${folder.id}`),
  );
  await dialog.getByRole("button", { name: "Save Changes" }).click();
  const response = await saved;
  expect(response.ok(), `saving the folder: ${response.status()}`).toBe(true);
  expect(
    response.request().postDataJSON(),
    "saving should send every edit",
  ).toEqual({ name: edit.name, color: FOLDER_COLOR[edit.color], icon });
  await expect(dialog).toBeHidden();
  return (await response.json()) as Folder;
}

/** Opens Edit folder and checks it shows what the folder was saved with,
 * then closes it unchanged. */
export async function expectEditDialogShows(
  page: Page,
  folder: Folder,
  colorLabel: FolderColorLabel,
) {
  await folderCard(page, folder.id)
    .getByRole("button", { name: "Edit folder" })
    .click();
  const dialog = page.getByRole("dialog", { name: "Edit Folder" });
  await expect(
    dialog.getByRole("textbox", { name: "Folder name" }),
  ).toHaveValue(folder.name);
  await expect(
    dialog.getByRole("combobox", { name: "Folder color" }),
  ).toContainText(colorLabel);
  await expect(
    dialog.getByText("Selected:").locator("xpath=following-sibling::div[1]"),
  ).toHaveText(folder.icon ?? "");
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
}

export async function deleteFolderInDialog(page: Page, folder: Folder) {
  await folderCard(page, folder.id)
    .getByRole("button", { name: "Delete folder" })
    .click();
  const dialog = page.getByRole("dialog", { name: "Delete folder" });
  await expect(
    dialog,
    "the dialog should say what becomes of the folder's workflows",
  ).toContainText(
    "Workflows inside this folder will be moved back to your library.",
  );
  const deleted = page.waitForResponse(
    isCall("DELETE", `/api/library/folders/${folder.id}`),
  );
  await dialog.getByRole("button", { name: "Delete Folder" }).click();
  expect((await deleted).status(), "deleting the folder").toBe(204);
  await expect(dialog).toBeHidden();
  await expect(folderCard(page, folder.id)).toHaveCount(0);
}

/** Move to folder, from the workflow card's menu. */
export async function moveInDialog(
  page: Page,
  workflow: Workflow,
  folder: Folder,
) {
  await openMoveDialog(page, workflow);
  const dialog = moveDialog(page);
  const moved = page.waitForResponse(
    isCall("POST", "/api/library/folders/agents/bulk-move"),
  );
  await dialog.getByRole("button").filter({ hasText: folder.name }).click();
  await expectMoved(await moved, workflow, folder.id);
  await expect(dialog).toBeHidden();
}

export async function openMoveDialog(page: Page, workflow: Workflow) {
  await openCardMenu(page, workflow);
  await page.getByRole("menuitem", { name: "Move to folder" }).click();
  await expect(moveDialog(page)).toBeVisible();
}

export function moveDialog(page: Page) {
  return page.getByRole("dialog", { name: "Move to folder" });
}

/** Remove from folder, from the workflow card's menu. */
export async function removeFromFolder(page: Page, workflow: Workflow) {
  await openCardMenu(page, workflow);
  const moved = page.waitForResponse(
    isCall("POST", "/api/library/folders/agents/bulk-move"),
  );
  await page.getByRole("menuitem", { name: "Remove from folder" }).click();
  await expectMoved(await moved, workflow, null);
}

async function openCardMenu(page: Page, workflow: Workflow) {
  await workflowCard(page, workflow)
    .getByRole("button", { name: "More actions" })
    .click();
}

async function expectMoved(
  response: Response,
  workflow: Workflow,
  folderID: string | null,
) {
  expect(response.ok(), `moving ${workflow.name}: ${response.status()}`).toBe(
    true,
  );
  expect(response.request().postDataJSON()).toEqual({
    agent_ids: [workflow.libraryAgentID],
    folder_id: folderID,
  });
  const moved = (await response.json()) as LibraryEntry[];
  expect(
    moved.map((entry) => [entry.id, entry.folder_id]),
    "the move should report the workflow it moved, where it moved it",
  ).toEqual([[workflow.libraryAgentID, folderID]]);
}

export async function openFolder(page: Page, folder: Folder) {
  await folderCard(page, folder.id).getByTestId("library-folder-name").click();
  await expect(page).toHaveURL(new RegExp(`[?&]folder=${folder.id}`));
  await expect(breadcrumb(page)).toBeVisible();
}

/** "My Library / <folder>", above an open folder. */
export function breadcrumb(page: Page) {
  return page
    .getByRole("button", { name: "My Library", exact: true })
    .locator("xpath=..");
}

export async function backToLibrary(page: Page) {
  await page.getByRole("button", { name: "My Library", exact: true }).click();
  await expect(page).not.toHaveURL(/[?&]folder=/);
  await expect(allTab(page)).toBeVisible();
}

export async function setFavorite(
  page: Page,
  workflow: Workflow,
  favorite: boolean,
) {
  const card = workflowCard(page, workflow);
  // The heart shows on hover until the workflow is a favorite.
  await card.hover();
  const saved = page.waitForResponse(
    isCall("PATCH", `/api/library/agents/${workflow.libraryAgentID}`),
  );
  await card
    .getByRole("button", {
      name: favorite ? "Add to favorites" : "Remove from favorites",
    })
    .click();
  const response = await saved;
  expect(response.ok(), `saving the favorite: ${response.status()}`).toBe(true);
  expect(response.request().postDataJSON()).toEqual({ is_favorite: favorite });
}

export async function readFavoritesCount(page: Page) {
  const response = await page.request.get(
    "/api/proxy/api/library/agents/favorites",
  );
  expect(response.ok(), `listing favorites: ${response.status()}`).toBe(true);
  return ((await response.json()) as { pagination: { total_items: number } })
    .pagination.total_items;
}

export function searchBox(page: Page) {
  return page.getByRole("textbox", { name: "Search workflows" });
}

/** Types a search and waits for the list to ask for it. */
export async function search(page: Page, term: string) {
  const asked = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return (
      url.pathname.endsWith("/api/library/agents") &&
      url.searchParams.get("search_term") === term
    );
  });
  await searchBox(page).fill(term);
  await asked;
}

/** Clears the search and waits for something only the unsearched list
 * shows: that list may come from the cache, with no request to wait on. */
export async function clearSearch(page: Page, unsearched: Locator) {
  await searchBox(page).fill("");
  await expect(searchBox(page)).toHaveValue("");
  await expect(unsearched).toBeVisible();
}

export function sortMenu(page: Page) {
  return page.getByTestId("sort-by-dropdown").getByRole("combobox");
}

export async function sortBy(page: Page, label: SortLabel) {
  await sortMenu(page).click();
  await page.getByRole("option", { name: label, exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`[?&]sort=${SORTS[label]}\\b`));
}

/** Ends the session and starts a new one, with no page of the app open
 * meanwhile: /logout's own requests race its sign-out (templates journey). */
export async function signOutAndBackIn(page: Page, user: JourneyUser) {
  await page.goto("about:blank");
  const signedOut = await page.request.post("/api/auth/sign-out", {
    headers: authHeaders(),
    data: {},
  });
  expect(signedOut.ok(), `signing out: ${signedOut.status()}`).toBe(true);
  const refused = await page.request.get("/api/proxy/api/library/folders");
  expect(
    refused.status(),
    "signed out, the folders should be out of reach",
  ).toBe(401);
  await signIn(page, user);
}

/** What the journey saw where the product's intent is an open question:
 * shown in the report beside the assertions, never asserted. */
export function observe(testInfo: TestInfo, description: string) {
  testInfo.annotations.push({ type: "observed", description });
}

function isCall(method: string, path: string) {
  return (response: Response) =>
    response.request().method() === method &&
    new URL(response.url()).pathname.endsWith(path);
}

async function pickColor(page: Page, dialog: Locator, label: string) {
  await dialog.getByRole("combobox", { name: "Folder color" }).click();
  await page.getByRole("option", { name: label, exact: true }).click();
}

// The picker's emoji at `index`, which it returns.
async function pickIcon(dialog: Locator, index: number) {
  const emoji = dialog.locator("button[data-emoji]").nth(index);
  await expect(emoji, "the emoji picker should load").toBeVisible();
  const icon = await emoji.getAttribute("data-emoji");
  expect(icon, "an emoji button names its emoji").toBeTruthy();
  await emoji.click();
  await expect(
    dialog.getByText("Selected:").locator("xpath=following-sibling::div[1]"),
    "the dialog should show the picked emoji",
  ).toHaveText(icon!);
  return icon!;
}

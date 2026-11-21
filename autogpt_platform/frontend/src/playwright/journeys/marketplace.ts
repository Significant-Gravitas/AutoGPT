import { randomBytes } from "crypto";
import path from "path";
import {
  expect,
  type Locator,
  type Page,
  type Response,
} from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import {
  assertRunOutputValue,
  getActiveItemId,
  getRunStatus as readStatusBadge,
  waitForAgentPageLoad,
} from "../pages/library.page";

// The marketplace journey's steps: the agent each test builds, and what its
// creator, the admin and the shopper do with its listing.

const AGENT_INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b";
const CALCULATOR_BLOCK_ID = "b1ab9b19-67a6-406d-abf5-2dba76d00c79";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";
const INPUT_NAME = "number";
const OUTPUT_NAME = "total";
const RUN_INPUT = 8;
const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;
// A real PNG, so the listing's thumbnail goes through the real upload.
const THUMBNAIL = path.join(__dirname, "../assets/test-thumbnail.png");

// What one version of the agent does to the number it is given. Versions do
// different arithmetic, so a run's output names the version that ran.
interface Arithmetic {
  operation: "Add" | "Multiply";
  operand: number;
  // What a run with RUN_INPUT outputs.
  output: number;
}

export const ADD_34: Arithmetic = { operation: "Add", operand: 34, output: 42 };
export const TIMES_3: Arithmetic = {
  operation: "Multiply",
  operand: 3,
  output: 24,
};

interface Agent {
  graphID: string;
  version: number;
  name: string;
}

interface Listing {
  title: string;
  slug: string;
  tagline: string;
  description: string;
}

interface Submission {
  listing_version_id: string;
  graph_id: string;
  graph_version: number;
  name: string;
  slug: string;
  status: string;
  review_comments?: string | null;
}

interface AdminListing {
  slug: string;
  active_listing_version_id: string | null;
  has_approved_version: boolean;
  creator_email: string | null;
  versions: (Submission & { internal_comments: string | null })[];
}

interface StoreAgent {
  slug: string;
  agent_name: string;
  creator: string;
}

interface LibraryAgent {
  id: string;
  graph_id: string;
  graph_version: number;
}

interface Run {
  status: string;
  graph_version: number;
  outputs: Record<string, unknown[]>;
}

// A made-up word. The store's search matches whole words, so a word no other
// listing has finds this listing alone.
export function uniqueWord() {
  return Array.from(randomBytes(10), (byte) =>
    String.fromCharCode(97 + (byte % 26)),
  ).join("");
}

export function newListing(): Listing {
  const word = uniqueWord();
  return {
    title: `Journey listing ${word}`,
    slug: `journey-${word}`,
    tagline: "Does arithmetic on any number you give it",
    description: `Does arithmetic on a number, for the marketplace journey (${word}).`,
  };
}

export async function readProfile(page: Page) {
  const response = await page.request.get("/api/proxy/api/store/profile");
  expect(response.ok(), `reading the profile: ${response.status()}`).toBe(true);
  return (await response.json()) as { username: string; name: string };
}

export async function createAgent(
  page: Page,
  arithmetic: Arithmetic,
): Promise<Agent> {
  const name = createUniqueAgentName("Journey marketplace");
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: { source: "upload", graph: agentGraph(name, arithmetic) },
  });
  expect(response.ok(), `creating the agent: ${response.status()}`).toBe(true);
  const graph = (await response.json()) as { id: string; version: number };
  return { graphID: graph.id, version: graph.version, name };
}

// Saves the agent doing other arithmetic, as its creator's next version.
export async function saveVersion(
  page: Page,
  agent: Agent,
  arithmetic: Arithmetic,
): Promise<Agent> {
  const response = await page.request.put(
    `/api/proxy/api/graphs/${agent.graphID}`,
    { data: { id: agent.graphID, ...agentGraph(agent.name, arithmetic) } },
  );
  expect(response.ok(), `saving the edit: ${response.status()}`).toBe(true);
  const { graph } = (await response.json()) as { graph: { version: number } };
  expect(graph.version, "the edit should be the agent's next version").toBe(
    agent.version + 1,
  );
  return { ...agent, version: graph.version };
}

// AgentInput "number" -> Calculator -> AgentOutput "total".
function agentGraph(name: string, { operation, operand }: Arithmetic) {
  return {
    name,
    description: "Does arithmetic on a number, for the marketplace journey",
    is_active: true,
    nodes: [
      node("number", AGENT_INPUT_BLOCK_ID, { name: INPUT_NAME }, [0, 0]),
      node(
        "arithmetic",
        CALCULATOR_BLOCK_ID,
        { operation, b: operand, round_result: false },
        [400, 0],
      ),
      node("total", AGENT_OUTPUT_BLOCK_ID, { name: OUTPUT_NAME }, [800, 0]),
    ],
    links: [
      link("number", "result", "arithmetic", "a"),
      link("arithmetic", "result", "total", "value"),
    ],
  };
}

function node(
  id: string,
  blockID: string,
  input: Record<string, unknown>,
  [x, y]: [number, number],
) {
  return {
    id,
    block_id: blockID,
    input_default: input,
    metadata: { position: { x, y } },
    input_links: [],
    output_links: [],
  };
}

function link(source: string, output: string, sink: string, input: string) {
  return {
    source_id: source,
    source_name: output,
    sink_id: sink,
    sink_name: input,
  };
}

// The creator publishes the way the marketplace invites them to, thumbnail
// and all.
export async function submitInPublishDialog(
  page: Page,
  agent: Agent,
  listing: Listing,
): Promise<Submission> {
  await page.goto("/marketplace");
  const dialog = page.getByTestId("publish-agent-modal");
  await clickUntilOpen(
    page.getByRole("button", { name: "Publish your workflows" }),
    dialog,
  );
  await dialog
    .getByTestId("agent-card")
    .filter({ hasText: agent.name })
    .click();
  await dialog.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    dialog.getByText("Write a bit of details about your workflow"),
  ).toBeVisible();

  await dialog.getByLabel("Title").fill(listing.title);
  await dialog.getByLabel("Tagline").fill(listing.tagline);
  await dialog.getByLabel("Slug").fill(listing.slug);
  await dialog.getByRole("combobox", { name: "Category" }).click();
  await page.getByRole("option", { name: "Operations" }).click();

  await dialog.getByRole("button", { name: /Thumbnails/ }).click();
  const uploaded = page.waitForResponse(isMediaUpload);
  await dialog.locator("#image-upload").setInputFiles(THUMBNAIL);
  const upload = await uploaded;
  expect(upload.ok(), `uploading the thumbnail: ${upload.status()}`).toBe(true);
  const thumbnailURL = (await upload.json()) as string;
  await expect(dialog.getByTestId("thumbnail-remove-0")).toBeVisible();

  await dialog.getByRole("button", { name: /Experience details/ }).click();
  await dialog.getByLabel("Description").fill(listing.description);

  const submitted = page.waitForResponse(isSubmissionCreate);
  await dialog.getByRole("button", { name: "Submit for review" }).click();
  const response = await submitted;
  expect(response.ok(), `submitting for review: ${response.status()}`).toBe(
    true,
  );
  expect(
    response.request().postDataJSON(),
    "the dialog should submit the listing as filled in",
  ).toMatchObject({
    graph_id: agent.graphID,
    graph_version: agent.version,
    name: listing.title,
    slug: listing.slug,
    sub_heading: listing.tagline,
    description: listing.description,
    image_urls: expect.arrayContaining([thumbnailURL]),
  });
  await expect(dialog.getByText("Submission received")).toBeVisible();
  return (await response.json()) as Submission;
}

// Submits what the publish dialog would, for a step another test walks
// through the dialog. With `changesSummary`, it is an update.
export async function submitDirectly(
  page: Page,
  agent: Agent,
  listing: Listing,
  changesSummary?: string,
): Promise<Submission> {
  const response = await page.request.post("/api/proxy/api/store/submissions", {
    data: {
      graph_id: agent.graphID,
      graph_version: agent.version,
      name: listing.title,
      slug: listing.slug,
      sub_heading: listing.tagline,
      description: listing.description,
      categories: ["operations"],
      changes_summary: changesSummary,
    },
  });
  expect(response.ok(), `submitting for review: ${response.status()}`).toBe(
    true,
  );
  const submission = (await response.json()) as Submission;
  expect(submission, "the submission should wait for review").toMatchObject({
    graph_id: agent.graphID,
    graph_version: agent.version,
    status: "PENDING",
  });
  return submission;
}

// The admin looks the submission up by name on the review page, where it
// waits as Pending under its creator's email.
export async function openReviewRow(
  admin: Page,
  listing: Listing,
  creatorEmail: string,
) {
  await admin.goto(
    `/admin/marketplace?search=${encodeURIComponent(listing.title)}`,
  );
  const row = admin.getByRole("row").filter({ hasText: listing.title });
  await expect(row, "the review page should list the submission").toHaveCount(
    1,
  );
  await expect(row).toContainText(creatorEmail);
  await expect(row.getByText("Pending", { exact: true })).toBeVisible();
  return row;
}

export async function approveInReviewPage(
  admin: Page,
  listing: Listing,
  creatorEmail: string,
  comment: string,
) {
  const row = await openReviewRow(admin, listing, creatorEmail);
  const dialog = admin.getByRole("dialog", { name: "Approve Agent" });
  await clickUntilOpen(
    row.getByRole("button", { name: "Approve", exact: true }),
    dialog,
  );
  await dialog.getByLabel("Comments (Optional)").fill(comment);
  await submitReview(
    admin,
    dialog.getByRole("button", { name: "Approve", exact: true }),
  );
  await expect(
    row.getByText("Approved", { exact: true }),
    "the review page should show the listing approved",
  ).toBeVisible();
}

// The page is server-rendered, and a click that lands before it hydrates
// does nothing, so click again until the dialog shows.
export async function clickUntilOpen(trigger: Locator, dialog: Locator) {
  await expect(async () => {
    if (!(await dialog.isVisible())) await trigger.click({ timeout: 2_000 });
    await expect(dialog).toBeVisible({ timeout: 2_000 });
  }).toPass({ timeout: 20_000 });
}

// The decision goes to the backend through a server action, which reports a
// failure only to the console, where the guard sees it.
export async function submitReview(admin: Page, decide: Locator) {
  const acted = admin.waitForResponse(
    (response) =>
      response.request().method() === "POST" &&
      response.request().headers()["next-action"] !== undefined,
  );
  await decide.click();
  const response = await acted;
  expect(
    response.ok(),
    `the review's server action: ${response.status()}`,
  ).toBe(true);
}

// Approves what the review page would, for a step another test walks
// through the page.
export async function approveDirectly(admin: Page, submission: Submission) {
  const response = await admin.request.post(
    `/api/proxy/api/store/admin/submissions/${submission.listing_version_id}/review`,
    {
      data: {
        store_listing_version_id: submission.listing_version_id,
        is_approved: true,
        comments: "Approved for the marketplace journey",
      },
    },
  );
  expect(response.ok(), `approving: ${response.status()}`).toBe(true);
}

export async function readAdminListing(
  admin: Page,
  listing: Listing,
): Promise<AdminListing> {
  const response = await admin.request.get(
    "/api/proxy/api/store/admin/listings",
    { params: { search: listing.title } },
  );
  expect(
    response.ok(),
    `reading the admin listings: ${response.status()}`,
  ).toBe(true);
  const { listings } = (await response.json()) as { listings: AdminListing[] };
  expect(
    listings.map((found) => found.slug),
    "the admin listings should hold the listing once",
  ).toEqual([listing.slug]);
  return listings[0];
}

export async function readMySubmissions(page: Page) {
  const response = await page.request.get("/api/proxy/api/store/submissions");
  expect(
    response.ok(),
    `reading the creator's submissions: ${response.status()}`,
  ).toBe(true);
  const body = await response.text();
  const parsed = (JSON.parse(body) as { submissions: Submission[] })
    .submissions;
  return { body, parsed };
}

export async function readStoreAgent(
  page: Page,
  creator: string,
  slug: string,
) {
  const response = await page.request.get(
    `/api/proxy/api/store/agents/${encodeURIComponent(creator)}/${encodeURIComponent(slug)}`,
  );
  expect(response.ok(), `reading the listing: ${response.status()}`).toBe(true);
  return (await response.json()) as { store_listing_version_id: string };
}

// What the marketplace's search page shows for `term`, as the store answered
// the page's own request, once the page shows its results. With embeddings
// on, the store also returns listings that only resemble the term, so check
// for one listing rather than the whole answer.
async function searchMarketplace(page: Page, term: string) {
  const answered = page.waitForResponse(isStoreSearch(term));
  await page.goto(`/marketplace/search?searchTerm=${encodeURIComponent(term)}`);
  const response = await answered;
  expect(response.ok(), `searching the marketplace: ${response.status()}`).toBe(
    true,
  );
  await expect(
    page.getByRole("heading", { name: "Showing results for:" }),
  ).toBeVisible();
  return ((await response.json()) as { agents: StoreAgent[] }).agents;
}

function storeCard(page: Page, listing: Listing) {
  return page
    .locator('[data-testid="store-card"]:visible')
    .filter({ hasText: listing.title });
}

// Someone searching for the listing doesn't find it: the store leaves it out
// of its answer, and the page shows no card for it.
export async function expectNotInMarketplace(
  page: Page,
  listing: Listing,
  why: string,
) {
  const found = await searchMarketplace(page, listing.title);
  expect(
    found.map((agent) => agent.slug),
    why,
  ).not.toContain(listing.slug);
  await expect(storeCard(page, listing), why).toHaveCount(0);
}

// Someone searches the marketplace for the listing, finds it under its
// creator, and opens it.
export async function openListing(
  page: Page,
  listing: Listing,
  creatorUsername: string,
) {
  expect(
    await searchMarketplace(page, listing.title),
    "the approved listing should be in the marketplace, under its creator",
  ).toContainEqual(
    expect.objectContaining({
      slug: listing.slug,
      agent_name: listing.title,
      creator: creatorUsername,
    }),
  );
  const card = storeCard(page, listing);
  await expect(card).toContainText(`by ${creatorUsername}`);
  await card.click();
  await expect(page).toHaveURL(
    new RegExp(
      `/marketplace/agent/${escapeRegex(creatorUsername)}/${listing.slug}$`,
    ),
  );
  await expect(page.getByTestId("agent-title")).toHaveText(listing.title);
  await expect(page.getByTestId("agent-creator")).toContainText(
    creatorUsername,
  );
}

export async function addToLibrary(
  page: Page,
  submission: Submission,
): Promise<LibraryAgent> {
  const added = page.waitForResponse(isLibraryAdd);
  await page.getByTestId("agent-add-library-button").click();
  const response = await added;
  expect(response.ok(), `adding to the Library: ${response.status()}`).toBe(
    true,
  );
  expect(
    response.request().postDataJSON(),
    "the listing page should add the version the marketplace serves",
  ).toEqual({ store_listing_version_id: submission.listing_version_id });
  const libraryAgent = (await response.json()) as LibraryAgent;
  await expect(page).toHaveURL(
    new RegExp(`/library/agents/${libraryAgent.id}`),
  );
  // A new agent's page settles on its empty state once its run list has
  // loaded; until then it can still be a placeholder.
  await waitForAgentPageLoad(page);
  await expect(
    page.getByRole("button", { name: "Setup your task" }),
  ).toBeVisible();
  return libraryAgent;
}

// Runs the agent the way a person does from its new Library page, and leaves
// the page on the run view it opens.
export async function runFromLibrary(page: Page, libraryAgent: LibraryAgent) {
  await page.getByRole("button", { name: "Setup your task" }).click();
  const runDialog = page.getByRole("dialog", { name: "Run Workflow" });
  await runDialog
    .getByTestId(`agent-input-${INPUT_NAME}`)
    .fill(String(RUN_INPUT));
  const started = page.waitForResponse(isRunStart(libraryAgent.graph_id));
  await runDialog
    .getByRole("button", { name: "Start Task", exact: true })
    .click();
  const response = await started;
  expect(response.ok(), `starting the run: ${response.status()}`).toBe(true);
  expect(
    new URL(response.url()).pathname,
    "the run should start the version in the Library",
  ).toMatch(
    new RegExp(
      `/graphs/${libraryAgent.graph_id}/execute/${libraryAgent.graph_version}$`,
    ),
  );
  const { id } = (await response.json()) as { id: string };
  await expect
    .poll(() => getActiveItemId(page), {
      message: "the run view should open on the new run",
    })
    .toBe(id);
  return id;
}

// The run completes as the Library's version of the agent does its
// arithmetic, in the API and in the run view the page is on.
export async function expectRunOf(
  page: Page,
  libraryAgent: LibraryAgent,
  runID: string,
  arithmetic: Arithmetic,
) {
  const run = await waitForRun(page, libraryAgent.graph_id, runID);
  expect(run.status, "the run should complete").toBe("COMPLETED");
  expect(
    run,
    `the run should be version ${libraryAgent.graph_version}, which makes ` +
      `${RUN_INPUT} into ${arithmetic.output}`,
  ).toMatchObject({
    graph_version: libraryAgent.graph_version,
    outputs: { [OUTPUT_NAME]: [arithmetic.output] },
  });
  await expect
    .poll(() => readStatusBadge(page), {
      message: "the run view should show the run completed",
    })
    .toBe("completed");
  await assertRunOutputValue(
    page,
    OUTPUT_NAME,
    new RegExp(`^${arithmetic.output}(?:\\.0+)?$`),
  );
}

// Runs start INCOMPLETE, then QUEUED, so wait for one that has ended.
async function waitForRun(page: Page, graphID: string, runID: string) {
  await expect
    .poll(async () => (await readRun(page, graphID, runID)).status, {
      message: "the run should end",
      timeout: 30_000,
    })
    .toMatch(RUN_ENDED);
  return readRun(page, graphID, runID);
}

async function readRun(page: Page, graphID: string, runID: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${graphID}/executions/${runID}`,
  );
  expect(response.ok(), `reading the run: ${response.status()}`).toBe(true);
  return (await response.json()) as Run;
}

function escapeRegex(text: string) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function isMediaUpload(response: Response) {
  return (
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.endsWith("/api/store/submissions/media")
  );
}

function isSubmissionCreate(response: Response) {
  return (
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.endsWith("/api/store/submissions")
  );
}

function isStoreSearch(term: string) {
  return (response: Response) => {
    const url = new URL(response.url());
    return (
      response.request().method() === "GET" &&
      url.pathname.endsWith("/api/store/agents") &&
      url.searchParams.get("search_query") === term
    );
  };
}

function isLibraryAdd(response: Response) {
  return (
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.endsWith("/api/library/agents")
  );
}

function isRunStart(graphID: string) {
  return (response: Response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.includes(
      `/api/graphs/${graphID}/execute/`,
    );
}

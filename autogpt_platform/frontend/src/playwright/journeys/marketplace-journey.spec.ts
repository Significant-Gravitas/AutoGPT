import { randomBytes } from "crypto";
import path from "path";
import type { Browser, Locator, Page, Response } from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import {
  assertRunOutputValue,
  getActiveItemId,
  getRunStatus as readStatusBadge,
  waitForAgentPageLoad,
} from "../pages/library.page";
import { markChangelogSeen } from "../utils/changelog";
import { adminGrantUnavailable, grantAdminRole } from "./admin-role";
import { expect, serveHostedFlags, test as journeyTest } from "./fixtures";
import { watchForUnexpectedErrors } from "./guard";
import { type JourneyUser, provisionNewJourneyUser } from "./journey-user";
import { JOURNEY_USER_AGENT } from "./user-agent";

const AGENT_INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b";
const CALCULATOR_BLOCK_ID = "b1ab9b19-67a6-406d-abf5-2dba76d00c79";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";
const INPUT_NAME = "number";
const OUTPUT_NAME = "total";
const RUN_INPUT = 8;
const RUN_ENDED = /^(COMPLETED|FAILED|TERMINATED)$/;
// A real PNG, so the listing's thumbnail goes through the real upload.
const THUMBNAIL = path.join(__dirname, "../assets/test-thumbnail.png");

// What one version of the agent does to the number it is given. Each version
// does something else, so a run's output names the version that ran.
interface Arithmetic {
  operation: "Add" | "Multiply";
  operand: number;
  // What a run with RUN_INPUT outputs.
  output: number;
}

// What the creator submits for review...
const REVIEWED: Arithmetic = { operation: "Add", operand: 34, output: 42 };
// ...and an edit they save while it waits, which nobody reviews.
const UNREVIEWED: Arithmetic = {
  operation: "Multiply",
  operand: 3,
  output: 24,
};

interface Person {
  page: Page;
  user: JourneyUser;
}

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

interface Run {
  status: string;
  graph_version: number;
  outputs: Record<string, unknown[]>;
}

const test = journeyTest.extend<{ admin: Person; shopper: Person }>({
  // A platform admin, who reviews submissions, in a browser of their own.
  admin: async ({ browser, baseURL, flagOverrides }, provide) => {
    await asSomeoneElse(
      browser,
      { baseURL, flagOverrides, who: "the admin" },
      async (person) => {
        await grantAdminRole(person.page, person.user);
        await provide(person);
      },
    );
  },
  // Someone who neither made nor reviewed the listing, in a browser of their
  // own.
  shopper: async ({ browser, baseURL, flagOverrides }, provide) => {
    await asSomeoneElse(
      browser,
      { baseURL, flagOverrides, who: "the shopper" },
      provide,
    );
  },
});

test.describe("marketplace journey", () => {
  const noAdmin = adminGrantUnavailable();
  test.skip(Boolean(noAdmin), noAdmin);

  test("an approved listing is in the marketplace under its creator, and someone else adds it and runs the version that was reviewed", async ({
    page,
    journeyUser,
    admin,
    shopper,
  }) => {
    const creator = await readProfile(page);
    const agent = await createAgent(page, REVIEWED);
    const listing = newListing();

    const submission = await submitInPublishDialog(page, agent, listing);
    expect(
      submission,
      "the submission should wait for review, with the agent as saved",
    ).toMatchObject({
      graph_id: agent.graphID,
      graph_version: agent.version,
      status: "PENDING",
    });

    // The creator keeps working while the listing waits. The marketplace
    // must still hand out what the admin reviewed, not the latest edit.
    await saveVersion(page, agent, UNREVIEWED);

    // Nobody else finds a listing before it is approved.
    expect(
      await searchMarketplace(shopper.page, listing.title),
      "a pending listing should not be in the marketplace",
    ).toEqual([]);
    await expect(shopper.page.getByText("No results found")).toBeVisible();

    const note = `Approved by the marketplace journey (${uniqueWord()})`;
    const row = await openReviewRow(admin.page, listing, journeyUser.email);
    await row.getByRole("button", { name: "Approve", exact: true }).click();
    const approveDialog = admin.page.getByRole("dialog", {
      name: "Approve Agent",
    });
    await approveDialog.getByLabel("Comments (Optional)").fill(note);
    await submitReview(
      admin.page,
      approveDialog.getByRole("button", { name: "Approve", exact: true }),
    );
    await expect(
      row.getByText("Approved", { exact: true }),
      "the admin page should show the listing approved",
    ).toBeVisible();
    const reviewed = await readAdminListing(admin.page, listing);
    expect(
      reviewed,
      "the store should serve the version the admin approved",
    ).toMatchObject({
      active_listing_version_id: submission.listing_version_id,
      has_approved_version: true,
      creator_email: journeyUser.email,
      versions: [
        {
          listing_version_id: submission.listing_version_id,
          graph_version: agent.version,
          status: "APPROVED",
          review_comments: note,
        },
      ],
    });

    // Someone else finds it in the marketplace under its creator. The same
    // search came back empty before the approval, so this also proves the
    // approval dropped the store's cached answer.
    expect(
      await searchMarketplace(shopper.page, listing.title),
      "the approved listing should be in the marketplace, under its creator",
    ).toEqual([
      expect.objectContaining({
        slug: listing.slug,
        agent_name: listing.title,
        creator: creator.username,
      }),
    ]);
    const card = shopper.page
      .locator('[data-testid="store-card"]:visible')
      .filter({ hasText: listing.title });
    await expect(card).toContainText(`by ${creator.username}`);
    await card.click();
    await expect(shopper.page).toHaveURL(
      new RegExp(
        `/marketplace/agent/${escapeRegex(creator.username)}/${listing.slug}$`,
      ),
    );
    await expect(shopper.page.getByTestId("agent-title")).toHaveText(
      listing.title,
    );
    await expect(shopper.page.getByTestId("agent-creator")).toContainText(
      creator.username,
    );

    // Adding it gives them the reviewed version, which runs as reviewed.
    const libraryAgent = await addToLibrary(shopper.page, submission);
    expect(
      libraryAgent,
      "the shopper's copy should be the reviewed version of the agent",
    ).toMatchObject({ graph_id: agent.graphID, graph_version: agent.version });
    await expect(
      shopper.page.getByText(`by ${creator.name}`, { exact: true }),
      "the shopper's Library should credit the creator",
    ).toBeVisible();
    const runID = await runFromLibrary(shopper.page, libraryAgent);
    const run = await waitForRun(shopper.page, agent.graphID, runID);
    expect(run.status, "the shopper's run should complete").toBe("COMPLETED");
    expect(
      run,
      `the run should be the reviewed version (${RUN_INPUT} + ${REVIEWED.operand}), not the creator's later edit`,
    ).toMatchObject({
      graph_version: agent.version,
      outputs: { [OUTPUT_NAME]: [REVIEWED.output] },
    });
    await expect
      .poll(() => readStatusBadge(shopper.page), {
        message: "the run view should show the run completed",
      })
      .toBe("completed");
    await assertRunOutputValue(
      shopper.page,
      OUTPUT_NAME,
      new RegExp(`^${REVIEWED.output}(?:\\.0+)?$`),
    );
  });

  test("a rejected listing stays out of the marketplace, and its creator sees why", async ({
    page,
    journeyUser,
    admin,
    shopper,
  }) => {
    const agent = await createAgent(page, REVIEWED);
    const listing = newListing();
    // The approve journey submits through the publish dialog; this one is
    // about what a rejection does, so it submits directly.
    const submission = await submitDirectly(page, agent, listing);

    const reason = `Say what the workflow adds up (${uniqueWord()})`;
    const internalNote = `For admins only (${uniqueWord()})`;
    const row = await openReviewRow(admin.page, listing, journeyUser.email);
    await row.getByRole("button", { name: "Reject", exact: true }).click();
    const rejectDialog = admin.page.getByRole("dialog", {
      name: "Reject Agent",
    });
    await rejectDialog.getByLabel("Comments for Creator").fill(reason);
    await rejectDialog.getByLabel("Internal Comments").fill(internalNote);
    await submitReview(
      admin.page,
      rejectDialog.getByRole("button", { name: "Reject", exact: true }),
    );
    await expect(
      row.getByText("Rejected", { exact: true }),
      "the admin page should show the listing rejected",
    ).toBeVisible();
    expect(
      await readAdminListing(admin.page, listing),
      "the store should keep the rejection, with both comments",
    ).toMatchObject({
      has_approved_version: false,
      versions: [
        {
          listing_version_id: submission.listing_version_id,
          status: "REJECTED",
          review_comments: reason,
          internal_comments: internalNote,
        },
      ],
    });

    // The creator learns why, and nothing the admins kept to themselves.
    const mine = await readMySubmissions(page);
    expect(
      mine.parsed.find(
        (s) => s.listing_version_id === submission.listing_version_id,
      ),
      "the creator's submission should be rejected with the admin's reason",
    ).toMatchObject({ status: "REJECTED", review_comments: reason });
    expect(
      mine.body,
      "the creator should never get the admins' internal comment",
    ).not.toContain(internalNote);

    await page.goto("/settings/creator-dashboard");
    const submissionRow = page
      .getByTestId("submission-row")
      .filter({ hasText: listing.title });
    await expect(
      submissionRow.getByText("Needs changes", { exact: true }),
      "the creator's dashboard should show the submission rejected",
    ).toBeVisible();
    await submissionRow.getByTestId("submission-actions").click();
    await page.getByRole("menuitem", { name: "View submission" }).click();
    const details = page.getByTestId("publish-agent-modal");
    await expect(details.getByText("Review feedback")).toBeVisible();
    await expect(
      details.getByText(reason),
      "the creator should see the admin's reason",
    ).toBeVisible();
    await expect(
      details.getByText(internalNote),
      "the creator should not see the internal comment",
    ).toHaveCount(0);

    // Nobody else finds it, or can add it by its id.
    expect(
      await searchMarketplace(shopper.page, listing.title),
      "a rejected listing should not be in the marketplace",
    ).toEqual([]);
    await expect(shopper.page.getByText("No results found")).toBeVisible();
    const added = await shopper.page.request.post(
      "/api/proxy/api/library/agents",
      { data: { store_listing_version_id: submission.listing_version_id } },
    );
    expect(
      [403, 404],
      `adding a rejected listing by its id should be refused, got ${added.status()}`,
    ).toContain(added.status());
  });
});

// A brand-new user, signed in, in a browser context of their own that gets
// what the journey's context gets: its user agent and flags, the changelog
// marked seen, and the guard, whose findings fail the journey.
async function asSomeoneElse(
  browser: Browser,
  {
    baseURL,
    flagOverrides,
    who,
  }: {
    baseURL: string | undefined;
    flagOverrides: Record<string, unknown>;
    who: string;
  },
  body: (person: Person) => Promise<void>,
) {
  const context = await browser.newContext({
    baseURL,
    bypassCSP: true,
    userAgent: JOURNEY_USER_AGENT,
  });
  const guard = watchForUnexpectedErrors(context, []);
  try {
    await markChangelogSeen(context);
    await serveHostedFlags(context, flagOverrides);
    const page = await context.newPage();
    await body({ page, user: await provisionNewJourneyUser(page) });
  } catch (error) {
    await context.close();
    throw error;
  }
  const problems = await guard.collect();
  await context.close();
  expect(
    problems,
    `${who} hit ${problems.length} unexpected error(s):\n\n` +
      problems.join("\n\n"),
  ).toEqual([]);
}

// A made-up word. The store's search matches whole words, so a word no other
// listing has finds this listing alone.
function uniqueWord() {
  return Array.from(randomBytes(10), (byte) =>
    String.fromCharCode(97 + (byte % 26)),
  ).join("");
}

function newListing(): Listing {
  const word = uniqueWord();
  return {
    title: `Journey listing ${word}`,
    slug: `journey-${word}`,
    tagline: "Adds 34 to any number you give it",
    description: `Adds 34 to a number, for the marketplace journey (${word}).`,
  };
}

async function readProfile(page: Page) {
  const response = await page.request.get("/api/proxy/api/store/profile");
  expect(response.ok(), `reading the profile: ${response.status()}`).toBe(true);
  return (await response.json()) as { username: string; name: string };
}

async function createAgent(page: Page, arithmetic: Arithmetic): Promise<Agent> {
  const name = createUniqueAgentName("Journey marketplace");
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: { source: "upload", graph: agentGraph(name, arithmetic) },
  });
  expect(response.ok(), `creating the agent: ${response.status()}`).toBe(true);
  const graph = (await response.json()) as { id: string; version: number };
  return { graphID: graph.id, version: graph.version, name };
}

// Saves the agent doing other arithmetic, as its creator's next version.
async function saveVersion(page: Page, agent: Agent, arithmetic: Arithmetic) {
  const response = await page.request.put(
    `/api/proxy/api/graphs/${agent.graphID}`,
    { data: { id: agent.graphID, ...agentGraph(agent.name, arithmetic) } },
  );
  expect(response.ok(), `saving the edit: ${response.status()}`).toBe(true);
  const { graph } = (await response.json()) as { graph: { version: number } };
  expect(graph.version, "the edit should be the agent's next version").toBe(
    agent.version + 1,
  );
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
async function submitInPublishDialog(
  page: Page,
  agent: Agent,
  listing: Listing,
): Promise<Submission> {
  await page.goto("/marketplace");
  await page.getByRole("button", { name: "Publish your workflows" }).click();
  const dialog = page.getByTestId("publish-agent-modal");
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

async function submitDirectly(
  page: Page,
  agent: Agent,
  listing: Listing,
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
    },
  });
  expect(response.ok(), `submitting for review: ${response.status()}`).toBe(
    true,
  );
  const submission = (await response.json()) as Submission;
  expect(submission.status, "the submission should wait for review").toBe(
    "PENDING",
  );
  return submission;
}

// The admin looks the submission up by name on the review page, where it
// waits as Pending under its creator's email.
async function openReviewRow(
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

// The decision goes to the backend through a server action, which reports a
// failure only to the console, where the guard sees it.
async function submitReview(admin: Page, decide: Locator) {
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

async function readAdminListing(
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

async function readMySubmissions(page: Page) {
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

// What the marketplace's search page shows for `term`, as the store answered
// the page's own request.
async function searchMarketplace(page: Page, term: string) {
  const answered = page.waitForResponse(isStoreSearch(term));
  await page.goto(`/marketplace/search?searchTerm=${encodeURIComponent(term)}`);
  const response = await answered;
  expect(response.ok(), `searching the marketplace: ${response.status()}`).toBe(
    true,
  );
  return ((await response.json()) as { agents: StoreAgent[] }).agents;
}

async function addToLibrary(page: Page, submission: Submission) {
  const added = page.waitForResponse(isLibraryAdd);
  await page.getByTestId("agent-add-library-button").click();
  const response = await added;
  expect(response.ok(), `adding to the Library: ${response.status()}`).toBe(
    true,
  );
  expect(
    response.request().postDataJSON(),
    "the listing page should add the approved version",
  ).toEqual({ store_listing_version_id: submission.listing_version_id });
  const libraryAgent = (await response.json()) as {
    id: string;
    graph_id: string;
    graph_version: number;
  };
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
async function runFromLibrary(
  page: Page,
  libraryAgent: { id: string; graph_id: string; graph_version: number },
) {
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

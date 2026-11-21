import type { Browser, Page } from "@playwright/test";
import { markChangelogSeen } from "../utils/changelog";
import {
  adminGrantUnavailable,
  grantAdminRole,
  revokeAdminRole,
} from "./admin-role";
import { expect, serveHostedFlags, test as journeyTest } from "./fixtures";
import { watchForUnexpectedErrors } from "./guard";
import { type JourneyUser, provisionNewJourneyUser } from "./journey-user";
import {
  ADD_34,
  addToLibrary,
  approveDirectly,
  approveInReviewPage,
  clickUntilOpen,
  createAgent,
  expectNotInMarketplace,
  expectRunOf,
  newListing,
  openListing,
  openReviewRow,
  readAdminListing,
  readMySubmissions,
  readProfile,
  readStoreAgent,
  runFromLibrary,
  saveVersion,
  submitDirectly,
  submitInPublishDialog,
  submitReview,
  TIMES_3,
  uniqueWord,
} from "./marketplace";
import { JOURNEY_USER_AGENT } from "./user-agent";

interface Person {
  page: Page;
  user: JourneyUser;
}

const test = journeyTest.extend<{ admin: Person; shopper: Person }>({
  // A platform admin, who reviews submissions, in a browser of their own.
  admin: async ({ browser, baseURL, flagOverrides }, provide) => {
    await asSomeoneElse(
      browser,
      { baseURL, flagOverrides, who: "the admin" },
      async (person) => {
        try {
          await grantAdminRole(person.page, person.user);
          await provide(person);
        } finally {
          await revokeAdminRole(person.user);
        }
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
  // Three people and two reviews: the approve test takes about 50 s in CI,
  // so a slower runner needs more than the journeys' default 90 s.
  test.describe.configure({ timeout: 180_000 });

  test("an approved listing is in the marketplace under its creator, and someone else adds it and runs the version that was reviewed", async ({
    page,
    journeyUser,
    admin,
    shopper,
  }) => {
    const creator = await readProfile(page);
    const agent = await createAgent(page, ADD_34);
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
    await saveVersion(page, agent, TIMES_3);

    // Nobody else finds a listing before it is approved.
    await expectNotInMarketplace(
      shopper.page,
      listing,
      "a pending listing should not be in the marketplace",
    );

    const note = `Approved by the marketplace journey (${uniqueWord()})`;
    await approveInReviewPage(admin.page, listing, journeyUser.email, note);
    expect(
      await readAdminListing(admin.page, listing),
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

    // Someone else finds it under its creator. The same search came back
    // without it before the approval, so this also checks that the approval
    // dropped the store's cached answer.
    await openListing(shopper.page, listing, creator.username);
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
    await expectRunOf(shopper.page, libraryAgent, runID, ADD_34);
  });

  test("a rejected listing stays out of the marketplace, and its creator sees why", async ({
    page,
    journeyUser,
    admin,
    shopper,
  }) => {
    const agent = await createAgent(page, ADD_34);
    const listing = newListing();
    // The first test submits through the publish dialog; this one is about
    // what a rejection does, so it submits directly.
    const submission = await submitDirectly(page, agent, listing);

    const reason = `Say what the workflow adds up (${uniqueWord()})`;
    const internalNote = `For admins only (${uniqueWord()})`;
    const row = await openReviewRow(admin.page, listing, journeyUser.email);
    const rejectDialog = admin.page.getByRole("dialog", {
      name: "Reject Agent",
    });
    await clickUntilOpen(
      row.getByRole("button", { name: "Reject", exact: true }),
      rejectDialog,
    );
    await rejectDialog.getByLabel("Comments for Creator").fill(reason);
    await rejectDialog.getByLabel("Internal Comments").fill(internalNote);
    await submitReview(
      admin.page,
      rejectDialog.getByRole("button", { name: "Reject", exact: true }),
    );
    await expect(
      row.getByText("Rejected", { exact: true }),
      "the review page should show the listing rejected",
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
    await expectNotInMarketplace(
      shopper.page,
      listing,
      "a rejected listing should not be in the marketplace",
    );
    const added = await shopper.page.request.post(
      "/api/proxy/api/library/agents",
      { data: { store_listing_version_id: submission.listing_version_id } },
    );
    expect(
      [403, 404],
      `adding a rejected listing by its id should be refused, got ${added.status()}`,
    ).toContain(added.status());
  });

  test("an approved update is what someone adding the listing from then on gets", async ({
    page,
    journeyUser,
    admin,
    shopper,
  }) => {
    const creator = await readProfile(page);
    const first = await createAgent(page, ADD_34);
    const listing = newListing();
    // The first test publishes and approves through the pages; here the
    // listing is already in the marketplace when the update comes in.
    const firstSubmission = await submitDirectly(page, first, listing);
    await approveDirectly(admin.page, firstSubmission);

    const changed = await saveVersion(page, first, TIMES_3);
    const update = await submitDirectly(
      page,
      changed,
      listing,
      "Multiplies by 3 instead of adding 34",
    );
    expect(
      update.listing_version_id,
      "the update should be a new version of the listing",
    ).not.toBe(firstSubmission.listing_version_id);
    // Until the update is approved, the marketplace hands out the first
    // version.
    expect(
      await readStoreAgent(shopper.page, creator.username, listing.slug),
      "the marketplace should offer the first version while the update waits",
    ).toMatchObject({
      store_listing_version_id: firstSubmission.listing_version_id,
    });

    const note = `Update approved by the marketplace journey (${uniqueWord()})`;
    await approveInReviewPage(admin.page, listing, journeyUser.email, note);
    const reviewed = await readAdminListing(admin.page, listing);
    expect(
      reviewed.active_listing_version_id,
      "the store should serve the approved update",
    ).toBe(update.listing_version_id);
    expect(
      reviewed.versions,
      "the update should be approved, and the first version stay approved " +
        "for those who already added it",
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          listing_version_id: firstSubmission.listing_version_id,
          graph_version: first.version,
          status: "APPROVED",
        }),
        expect.objectContaining({
          listing_version_id: update.listing_version_id,
          graph_version: changed.version,
          status: "APPROVED",
          review_comments: note,
        }),
      ]),
    );

    // Someone adding it now gets the update, and runs it.
    await openListing(shopper.page, listing, creator.username);
    const libraryAgent = await addToLibrary(shopper.page, update);
    expect(
      libraryAgent,
      "the shopper's copy should be the approved update",
    ).toMatchObject({
      graph_id: first.graphID,
      graph_version: changed.version,
    });
    const runID = await runFromLibrary(shopper.page, libraryAgent);
    await expectRunOf(shopper.page, libraryAgent, runID, TIMES_3);
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
  let failure: unknown;
  try {
    await markChangelogSeen(context);
    await serveHostedFlags(context, flagOverrides);
    const page = await context.newPage();
    await body({ page, user: await provisionNewJourneyUser(page) });
  } catch (error) {
    failure = error;
  }
  let problems: string[];
  try {
    problems = await guard.collect();
  } finally {
    await context.close();
  }
  if (failure) {
    // What the guard saw often explains why the setup failed.
    if (failure instanceof Error && problems.length > 0) {
      failure.message +=
        `\n\n${who} also hit ${problems.length} unexpected error(s):\n\n` +
        problems.join("\n\n");
    }
    throw failure;
  }
  expect(
    problems,
    `${who} hit ${problems.length} unexpected error(s):\n\n` +
      problems.join("\n\n"),
  ).toEqual([]);
}

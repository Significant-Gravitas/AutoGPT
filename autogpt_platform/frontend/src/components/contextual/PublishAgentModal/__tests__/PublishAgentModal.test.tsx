import { useState } from "react";
import { QueryClient } from "@tanstack/react-query";
import { describe, expect, onTestFinished, test, vi } from "vitest";

import { getGetV2GetMyLiveSubmissionQueryKey } from "@/app/api/__generated__/endpoints/store/store";
import {
  getGetV2GetMyAgentsMockHandler,
  getGetV2GetMyLiveSubmissionMockHandler,
  getGetV2GetMyLiveSubmissionMockHandler422,
  getGetV2ListMySubmissionsMockHandler,
  getGetV2ListStoreCategoriesMockHandler,
  getPostV2CreateStoreSubmissionMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MyUnpublishedAgent } from "@/app/api/__generated__/models/myUnpublishedAgent";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

vi.mock("@/components/contextual/CronScheduler/cron-scheduler-dialog", () => ({
  CronExpressionDialog: () => null,
}));

// The success screen's confetti draws on a canvas, which jsdom doesn't have.
vi.mock("@/components/molecules/Confetti/Confetti", () => ({
  Confetti: () => null,
}));

vi.mock(
  "@/components/contextual/PublishAgentModal/components/AgentInfoStep/components/ThumbnailImages",
  () => ({
    ThumbnailImages: () => <div data-testid="thumbnail-images-mock" />,
  }),
);

import { PublishAgentModal } from "../PublishAgentModal";
import type { PublishState } from "../usePublishAgentModal";

const LISTED_AGENT: MyUnpublishedAgent = {
  graph_id: "g1",
  graph_version: 2,
  agent_name: "Agent name",
  description: "Agent description",
  agent_image: null,
  last_edited: new Date("2026-11-01T00:00:00Z"),
  recommended_schedule_cron: null,
};

const UNLISTED_AGENT: MyUnpublishedAgent = {
  graph_id: "g2",
  graph_version: 1,
  agent_name: "Fresh agent",
  description: "Fresh description",
  agent_image: null,
  last_edited: new Date("2026-11-02T00:00:00Z"),
  recommended_schedule_cron: null,
};

const LIVE_LISTING: StoreSubmission = {
  listing_id: "listing-1",
  user_id: "user-1",
  slug: "listed-slug",
  listing_version_id: "listing-version-1",
  listing_version: 1,
  graph_id: "g1",
  graph_version: 1,
  name: "Listed name",
  sub_heading: "Listed subheading",
  description: "Listed description",
  instructions: null,
  categories: ["marketing"],
  image_urls: [],
  video_url: null,
  agent_output_demo_url: null,
  submitted_at: new Date("2026-10-01T00:00:00Z"),
  changes_summary: null,
  status: SubmissionStatus.APPROVED,
};

// The creator's 20 newest submissions, all for other workflows: the listed
// agent's approved one is on the second page.
const NEWER_SUBMISSIONS: StoreSubmission[] = Array.from(
  { length: 20 },
  (_, index) => ({
    ...LIVE_LISTING,
    listing_id: `other-listing-${index}`,
    listing_version_id: `other-version-${index}`,
    graph_id: `other-graph-${index}`,
    slug: `other-slug-${index}`,
    name: `Other workflow ${index}`,
    submitted_at: new Date("2026-11-10T00:00:00Z"),
  }),
);

function liveListingOf(graphId: string): StoreSubmission | null {
  return graphId === LISTED_AGENT.graph_id ? LIVE_LISTING : null;
}

function installHandlers() {
  const agents = [LISTED_AGENT, UNLISTED_AGENT];
  server.use(
    getGetV2GetMyAgentsMockHandler({
      agents,
      pagination: {
        total_items: agents.length,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV2ListMySubmissionsMockHandler({
      submissions: NEWER_SUBMISSIONS,
      pagination: {
        total_items: NEWER_SUBMISSIONS.length + 1,
        total_pages: 2,
        current_page: 1,
        page_size: 20,
      },
      stats: {
        total: NEWER_SUBMISSIONS.length + 1,
        approved: NEWER_SUBMISSIONS.length + 1,
        pending: 0,
        total_runs: 0,
        average_rating: null,
      },
    }),
    getGetV2GetMyLiveSubmissionMockHandler(({ params }) =>
      liveListingOf(String(params.graphId)),
    ),
    getGetV2ListStoreCategoriesMockHandler([
      {
        value: "marketing",
        label: "Marketing",
        description: "Reach and convert customers",
      },
    ]),
  );
}

// A lookup that answers only once the test lets it.
function holdLiveListingLookup() {
  let release = () => {};
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  server.use(
    getGetV2GetMyLiveSubmissionMockHandler(async ({ params }) => {
      await released;
      return liveListingOf(String(params.graphId));
    }),
  );
  onTestFinished(() => release());
  return release;
}

function renderDialog(dialog: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(dialog, { queryClient });
  return queryClient;
}

async function waitForLookupToLand(queryClient: QueryClient, graphId: string) {
  await waitFor(() =>
    expect(
      queryClient.getQueryState(getGetV2GetMyLiveSubmissionQueryKey(graphId))
        ?.status,
    ).toBe("success"),
  );
}

// Opened the way the Marketplace page opens it: controlled, on the picker.
function MarketplacePublishDialog({
  stateChanges = [],
}: {
  stateChanges?: PublishState[];
}) {
  const [publishState, setPublishState] = useState<PublishState>({
    isOpen: true,
    step: "select",
    submissionData: null,
  });
  function handleStateChange(state: PublishState) {
    stateChanges.push(state);
    setPublishState(state);
  }
  return (
    <>
      <button
        type="button"
        data-testid="page-closes-dialog"
        onClick={() =>
          setPublishState({
            isOpen: false,
            step: "select",
            submissionData: null,
          })
        }
      />
      <PublishAgentModal
        showTrigger={false}
        targetState={publishState}
        onStateChange={handleStateChange}
      />
    </>
  );
}

// Opened the way the builder opens it: on the listing form, for one workflow.
function PreScopedPublishDialog() {
  const [publishState, setPublishState] = useState<PublishState>({
    isOpen: true,
    step: "info",
    submissionData: null,
  });
  return (
    <PublishAgentModal
      showTrigger={false}
      targetState={publishState}
      onStateChange={setPublishState}
      preSelectedAgentId={LISTED_AGENT.graph_id}
      preSelectedAgentVersion={LISTED_AGENT.graph_version}
    />
  );
}

// Opened the way the Library opens it: a new targetState on every render.
function LibraryStylePublishDialog() {
  const [, setRenders] = useState(0);
  return (
    <>
      <button
        type="button"
        data-testid="page-rerenders"
        onClick={() => setRenders((renders) => renders + 1)}
      />
      <PublishAgentModal
        showTrigger={false}
        targetState={{ isOpen: true, step: "info", submissionData: null }}
        onStateChange={() => {}}
        preSelectedAgentId={LISTED_AGENT.graph_id}
        preSelectedAgentVersion={LISTED_AGENT.graph_version}
      />
    </>
  );
}

async function pickAgentAndContinue(agentName: string) {
  fireEvent.click(await screen.findByText(agentName));
  fireEvent.click(screen.getByRole("button", { name: "Continue" }));
}

function continueButton() {
  return screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement;
}

async function expectUpdateFormFromLiveListing() {
  expect(await screen.findByText("Describe the update")).toBeDefined();
  expect(screen.getByRole("textbox", { name: "What changed?" })).toBeDefined();
  expect(await screen.findByDisplayValue("Listed name")).toBeDefined();
  expect(screen.getByDisplayValue("Listed subheading")).toBeDefined();
  expect(screen.getByDisplayValue("listed-slug")).toBeDefined();
  await waitFor(() =>
    expect(screen.getByRole("combobox", { name: "Category" }).textContent).toBe(
      "Marketing",
    ),
  );
  expect(screen.queryByDisplayValue("Agent name")).toBeNull();
}

const LOOKUP_FAILED =
  /couldn.t check whether this workflow is already in the Marketplace/i;

describe("PublishAgentModal opened from the Marketplace picker", () => {
  test("opens the update form, filled from the live listing, for an agent whose approved submission is past the creator's first page of submissions", async () => {
    installHandlers();
    renderDialog(<MarketplacePublishDialog />);

    await pickAgentAndContinue("Agent name");

    await expectUpdateFormFromLiveListing();
  });

  test("opens the new-listing form, filled from the agent, for an agent with nothing live", async () => {
    installHandlers();
    renderDialog(<MarketplacePublishDialog />);

    await pickAgentAndContinue("Fresh agent");

    expect(await screen.findByText("Build the store listing")).toBeDefined();
    expect(await screen.findByDisplayValue("Fresh agent")).toBeDefined();
    expect(screen.getByDisplayValue("fresh-agent")).toBeDefined();
    expect(screen.queryByRole("textbox", { name: "What changed?" })).toBeNull();
  });

  test("holds Continue in a loading state while a slow lookup runs, then opens the update form", async () => {
    installHandlers();
    const releaseLookup = holdLiveListingLookup();
    renderDialog(<MarketplacePublishDialog />);

    await pickAgentAndContinue("Agent name");

    await waitFor(() => expect(continueButton().disabled).toBe(true));
    expect(screen.getByText("Choose a workflow")).toBeDefined();
    expect(screen.queryByText("Build the store listing")).toBeNull();
    expect(screen.queryByText("Describe the update")).toBeNull();

    releaseLookup();

    await expectUpdateFormFromLiveListing();
  });

  test("keeps the creator on the picker with an error when the lookup fails, and Continue tries again", async () => {
    installHandlers();
    server.use(
      getGetV2GetMyLiveSubmissionMockHandler422(undefined, { once: true }),
    );
    renderDialog(<MarketplacePublishDialog />);

    await pickAgentAndContinue("Agent name");

    expect(await screen.findByText(LOOKUP_FAILED)).toBeDefined();
    expect(screen.getByText("Choose a workflow")).toBeDefined();
    expect(continueButton().disabled).toBe(false);
    expect(screen.queryByText("Build the store listing")).toBeNull();

    fireEvent.click(continueButton());

    await expectUpdateFormFromLiveListing();
    expect(screen.queryByText(LOOKUP_FAILED)).toBeNull();
  });

  // The dialog reports a step change to its parent as it makes it, so these
  // read the reports rather than wait on a render that should never come.
  test("stays closed when a lookup lands after the creator closed the dialog", async () => {
    installHandlers();
    const releaseLookup = holdLiveListingLookup();
    const stateChanges: PublishState[] = [];
    const queryClient = renderDialog(
      <MarketplacePublishDialog stateChanges={stateChanges} />,
    );

    await pickAgentAndContinue("Agent name");
    await waitFor(() => expect(continueButton().disabled).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(screen.queryByTestId("publish-agent-modal")).toBeNull(),
    );

    releaseLookup();

    await waitForLookupToLand(queryClient, LISTED_AGENT.graph_id);
    expect(stateChanges.map(({ isOpen, step }) => ({ isOpen, step }))).toEqual([
      { isOpen: false, step: "select" },
    ]);
    expect(screen.queryByTestId("publish-agent-modal")).toBeNull();
  });

  test("stays closed when a lookup lands after the page closed the dialog", async () => {
    installHandlers();
    const releaseLookup = holdLiveListingLookup();
    const stateChanges: PublishState[] = [];
    const queryClient = renderDialog(
      <MarketplacePublishDialog stateChanges={stateChanges} />,
    );

    await pickAgentAndContinue("Agent name");
    await waitFor(() => expect(continueButton().disabled).toBe(true));
    fireEvent.click(screen.getByTestId("page-closes-dialog"));
    await waitFor(() =>
      expect(screen.queryByTestId("publish-agent-modal")).toBeNull(),
    );

    releaseLookup();

    await waitForLookupToLand(queryClient, LISTED_AGENT.graph_id);
    expect(stateChanges).toEqual([]);
    expect(screen.queryByTestId("publish-agent-modal")).toBeNull();
  });

  test("drops a lookup for a workflow the creator picked away from", async () => {
    installHandlers();
    const releaseLookup = holdLiveListingLookup();
    const stateChanges: PublishState[] = [];
    const queryClient = renderDialog(
      <MarketplacePublishDialog stateChanges={stateChanges} />,
    );

    await pickAgentAndContinue("Agent name");
    await waitFor(() => expect(continueButton().disabled).toBe(true));
    fireEvent.click(screen.getByText("Fresh agent"));
    await waitFor(() => expect(continueButton().disabled).toBe(false));

    releaseLookup();

    await waitForLookupToLand(queryClient, LISTED_AGENT.graph_id);
    expect(stateChanges).toEqual([]);
    expect(screen.getByText("Choose a workflow")).toBeDefined();
  });
});

describe("PublishAgentModal opened pre-scoped to a workflow", () => {
  test("waits for the lookup before building the form, then opens the update form", async () => {
    installHandlers();
    const releaseLookup = holdLiveListingLookup();
    renderDialog(<PreScopedPublishDialog />);

    expect(
      await screen.findByRole("status", {
        name: "Checking this workflow's Marketplace listing",
      }),
    ).toBeDefined();
    expect(screen.queryByText("Build the store listing")).toBeNull();

    releaseLookup();

    await expectUpdateFormFromLiveListing();
  });

  test("keeps the update form from one lookup while a page re-renders it with a new targetState each time", async () => {
    installHandlers();
    const lookups = vi.fn();
    server.use(
      getGetV2GetMyLiveSubmissionMockHandler(({ params }) => {
        lookups(params.graphId);
        return liveListingOf(String(params.graphId));
      }),
    );
    renderDialog(<LibraryStylePublishDialog />);
    await expectUpdateFormFromLiveListing();

    fireEvent.click(screen.getByTestId("page-rerenders"));
    fireEvent.click(screen.getByTestId("page-rerenders"));

    await expectUpdateFormFromLiveListing();
    expect(lookups).toHaveBeenCalledOnce();
  });

  test("stays on the success screen after a submit while a page re-renders it with a new targetState each time", async () => {
    installHandlers();
    server.use(
      getGetV2GetMyLiveSubmissionMockHandler({
        ...LIVE_LISTING,
        image_urls: ["https://example.com/listed-thumbnail.png"],
      }),
      getPostV2CreateStoreSubmissionMockHandler({
        ...LIVE_LISTING,
        listing_version_id: "listing-version-2",
        listing_version: 2,
        status: SubmissionStatus.PENDING,
      }),
    );
    renderDialog(<LibraryStylePublishDialog />);
    await expectUpdateFormFromLiveListing();
    fireEvent.change(screen.getByRole("textbox", { name: "What changed?" }), {
      target: { value: "Clearer instructions" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Submit for review" }));
    expect(await screen.findByText("Submission received")).toBeDefined();

    fireEvent.click(screen.getByTestId("page-rerenders"));
    fireEvent.click(screen.getByTestId("page-rerenders"));

    // The step strip shows on every step but the success screen. It sits
    // outside the animated step, so it changes in the render that changes the
    // step, before the form could animate back in.
    expect(screen.queryByRole("list", { name: "Publish progress" })).toBeNull();
    expect(screen.getByText("Submission received")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Submit for review" }),
    ).toBeNull();
  });

  test("shows an error instead of the new-listing form when the lookup fails, and tries again", async () => {
    installHandlers();
    server.use(
      getGetV2GetMyLiveSubmissionMockHandler422(undefined, { once: true }),
    );
    renderDialog(<PreScopedPublishDialog />);

    expect(await screen.findByText(LOOKUP_FAILED)).toBeDefined();
    expect(screen.queryByText("Build the store listing")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    await expectUpdateFormFromLiveListing();
  });
});

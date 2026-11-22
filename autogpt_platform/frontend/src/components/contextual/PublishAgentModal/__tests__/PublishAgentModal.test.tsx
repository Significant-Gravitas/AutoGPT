import { useState } from "react";
import { QueryClient } from "@tanstack/react-query";
import { describe, expect, test, vi } from "vitest";

import { getGetV2ListMySubmissionsQueryKey } from "@/app/api/__generated__/endpoints/store/store";
import {
  getGetV2GetMyAgentsMockHandler,
  getGetV2ListMySubmissionsMockHandler,
  getGetV2ListStoreCategoriesMockHandler,
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
      submissions: [LIVE_LISTING],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
      stats: {
        total: 1,
        approved: 1,
        pending: 0,
        total_runs: 0,
        average_rating: null,
      },
    }),
    getGetV2ListStoreCategoriesMockHandler([
      {
        value: "marketing",
        label: "Marketing",
        description: "Reach and convert customers",
      },
    ]),
  );
}

// Opened the way the Marketplace page opens it: controlled, on the picker.
function MarketplacePublishDialog() {
  const [publishState, setPublishState] = useState<PublishState>({
    isOpen: true,
    step: "select",
    submissionData: null,
  });
  return (
    <PublishAgentModal
      showTrigger={false}
      targetState={publishState}
      onStateChange={setPublishState}
    />
  );
}

async function pickAgentAndContinue(agentName: string) {
  installHandlers();
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(<MarketplacePublishDialog />, { queryClient });

  // A person takes longer to pick than the dialog takes to load the
  // creator's submissions; wait for them so the pick doesn't race the load.
  await waitFor(() =>
    expect(
      queryClient.getQueryState(getGetV2ListMySubmissionsQueryKey())?.status,
    ).toBe("success"),
  );

  fireEvent.click(await screen.findByText(agentName));
  fireEvent.click(screen.getByRole("button", { name: "Continue" }));
}

describe("PublishAgentModal opened from the Marketplace picker", () => {
  test("opens the update form, filled from the live listing, for an agent that is already listed", async () => {
    await pickAgentAndContinue("Agent name");

    expect(await screen.findByText("Describe the update")).toBeDefined();
    expect(
      screen.getByRole("textbox", { name: "What changed?" }),
    ).toBeDefined();
    expect(await screen.findByDisplayValue("Listed name")).toBeDefined();
    expect(screen.getByDisplayValue("Listed subheading")).toBeDefined();
    expect(screen.getByDisplayValue("listed-slug")).toBeDefined();
    await waitFor(() =>
      expect(
        screen.getByRole("combobox", { name: "Category" }).textContent,
      ).toBe("Marketing"),
    );
    expect(screen.queryByDisplayValue("Agent name")).toBeNull();
  });

  test("opens the new-listing form, filled from the agent, for an agent with no approved submission", async () => {
    await pickAgentAndContinue("Fresh agent");

    expect(await screen.findByText("Build the store listing")).toBeDefined();
    expect(await screen.findByDisplayValue("Fresh agent")).toBeDefined();
    expect(screen.getByDisplayValue("fresh-agent")).toBeDefined();
    expect(screen.queryByRole("textbox", { name: "What changed?" })).toBeNull();
  });
});

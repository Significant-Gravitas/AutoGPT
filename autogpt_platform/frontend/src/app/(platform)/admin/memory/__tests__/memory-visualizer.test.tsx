import { describe, expect, test, vi } from "vitest";
import userEvent from "@testing-library/user-event";

import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2GetCommunityRebuildStatusMockHandler200,
  getGetV2GetCommunityRebuildStatusResponseMock200,
  getGetV2GetDreamPassStatusMockHandler200,
  getGetV2GetDreamPassStatusResponseMock200,
  getGetV2GetGraphMockHandler200,
  getGetV2GetGraphResponseMock200,
  getGetV2GetMemoryOverviewMockHandler200,
  getGetV2GetMemoryOverviewResponseMock200,
  getPostV2RebuildCommunitiesMockHandler202,
  getPostV2RebuildCommunitiesResponseMock202,
  getPostV2TriggerDreamPassMockHandler202,
  getPostV2TriggerDreamPassResponseMock202,
} from "@/app/api/__generated__/endpoints/admin/admin.msw";

// react-force-graph-2d uses HTMLCanvas + window APIs and pulls in d3 at
// import time. In the jsdom test env neither exists, so swap it for a
// trivial stub so MemoryVisualizer's tree is renderable.
vi.mock("react-force-graph-2d", () => ({
  default: () => null,
}));

import { MemoryVisualizer } from "../components/MemoryVisualizer";

function setupBaseHandlers() {
  server.use(
    getGetV2GetMemoryOverviewMockHandler200({
      ...getGetV2GetMemoryOverviewResponseMock200(),
      user_id: "u-1",
      group_id: "g-1",
      entities: 12,
      episodes: 30,
      relates_to_edges: 25,
      mentions_edges: 30,
      communities: 0,
    }),
    getGetV2GetGraphMockHandler200({
      ...getGetV2GetGraphResponseMock200(),
      user_id: "u-1",
      group_id: "g-1",
      nodes: [],
      edges: [],
      truncated: false,
    }),
  );
}

describe("MemoryVisualizer — 202 + polling contract", () => {
  test("renders rebuild and dream buttons + the overview chip", async () => {
    setupBaseHandlers();
    render(<MemoryVisualizer />);

    expect(
      await screen.findByRole("button", { name: /rebuild communities/i }),
    ).toBeDefined();
    expect(
      await screen.findByRole("button", { name: /dream pass/i }),
    ).toBeDefined();
    await waitFor(() => {
      expect(
        screen.queryByText((c) => /12/.test(c) && /entit/i.test(c)),
      ).toBeDefined();
    });
  });

  test("clicking 'Rebuild communities' → 202 + polled status flips button label", async () => {
    setupBaseHandlers();
    server.use(
      getPostV2RebuildCommunitiesMockHandler202({
        ...getPostV2RebuildCommunitiesResponseMock202(),
        job_id: "job-rebuild-1",
        state: "queued",
      }),
      getGetV2GetCommunityRebuildStatusMockHandler200({
        ...getGetV2GetCommunityRebuildStatusResponseMock200(),
        job_id: "job-rebuild-1",
        kind: "rebuild",
        state: "running",
        current_phase: "rebuild",
      }),
    );
    render(<MemoryVisualizer />);
    const btn = await screen.findByRole("button", {
      name: /rebuild communities/i,
    });
    await userEvent.click(btn);

    // After the 202 lands, the active job id is set and the poll
    // sees state=running with current_phase=rebuild → label morphs.
    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: /rebuild…/i }),
      ).toBeDefined();
    });
  });

  test("clicking 'Dream pass' → 202 + polled status shows phase in label", async () => {
    setupBaseHandlers();
    server.use(
      getPostV2TriggerDreamPassMockHandler202({
        ...getPostV2TriggerDreamPassResponseMock202(),
        job_id: "job-dream-1",
        state: "queued",
      }),
      getGetV2GetDreamPassStatusMockHandler200({
        ...getGetV2GetDreamPassStatusResponseMock200(),
        job_id: "job-dream-1",
        kind: "dream_pass",
        state: "submitted",
        current_phase: "consolidate",
      }),
    );
    render(<MemoryVisualizer />);
    const btn = await screen.findByRole("button", { name: /dream pass/i });
    await userEvent.click(btn);

    await waitFor(() => {
      expect(
        screen.queryByRole("button", {
          name: /batch submitted \(consolidate\)/i,
        }),
      ).toBeDefined();
    });
  });
});

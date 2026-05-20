import { describe, expect, test, vi } from "vitest";
import userEvent from "@testing-library/user-event";

import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2GetGraphMockHandler200,
  getGetV2GetGraphResponseMock200,
  getGetV2GetMemoryOverviewMockHandler200,
  getGetV2GetMemoryOverviewResponseMock200,
  getPostV2RebuildCommunitiesMockHandler200,
  getPostV2RebuildCommunitiesResponseMock200,
  getPostV2TriggerDreamPassMockHandler200,
  getPostV2TriggerDreamPassResponseMock200,
} from "@/app/api/__generated__/endpoints/admin/admin.msw";

// react-force-graph-2d uses HTMLCanvas + window APIs and pulls in d3 at
// import time. In the jsdom test env neither exists, so swap it for a
// trivial stub so MemoryVisualizer's tree is renderable.
vi.mock("react-force-graph-2d", () => ({
  default: () => null,
}));

import { MemoryVisualizer } from "../components/MemoryVisualizer";

function setupHandlers() {
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
    getPostV2RebuildCommunitiesMockHandler200({
      ...getPostV2RebuildCommunitiesResponseMock200(),
      user_id: "u-1",
      skipped: false,
      elapsed_seconds: 1.2,
      communities_built: { total: 5 },
    }),
    getPostV2TriggerDreamPassMockHandler200({
      ...getPostV2TriggerDreamPassResponseMock200(),
      user_id: "u-1",
      pass_id: "p-test",
      skipped: false,
      consolidated_count: 3,
      proposal_count: 2,
      demotion_count: 1,
      elapsed_seconds: 4.2,
      summary_for_user:
        "Reviewed 30 recent episodes; consolidated 3 facts and proposed 2 new findings.",
    }),
  );
}

describe("MemoryVisualizer — admin dream button", () => {
  test("renders both rebuild and dream buttons in the ControlBar", async () => {
    setupHandlers();
    render(<MemoryVisualizer />);
    expect(await screen.findByRole("button", { name: /rebuild communities/i }))
      .toBeDefined();
    expect(await screen.findByRole("button", { name: /run dream pass/i }))
      .toBeDefined();
  });

  test("clicking 'Run dream pass' calls the dream endpoint and surfaces the result chip", async () => {
    setupHandlers();
    render(<MemoryVisualizer />);

    const dreamBtn = await screen.findByRole("button", {
      name: /run dream pass/i,
    });
    await userEvent.click(dreamBtn);

    // While the mutation is pending the button label flips to "Dreaming…"
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /dreaming…/i })).toBeDefined();
    });

    // After the response lands the last-dream chip shows the elapsed time
    // and per-phase counts.
    await waitFor(() => {
      const node = screen.queryByText((content) =>
        content.includes("last dream") && content.includes("w=3") &&
          content.includes("p=2") && content.includes("d=1"),
      );
      expect(node).toBeDefined();
    });
  });

  test("dream skip result surfaces a skip reason chip", async () => {
    server.use(
      getGetV2GetMemoryOverviewMockHandler200(
        getGetV2GetMemoryOverviewResponseMock200(),
      ),
      getGetV2GetGraphMockHandler200(getGetV2GetGraphResponseMock200()),
      getPostV2TriggerDreamPassMockHandler200({
        ...getPostV2TriggerDreamPassResponseMock200(),
        skipped: true,
        skip_reason: "no_input",
        elapsed_seconds: 0.05,
      }),
    );
    render(<MemoryVisualizer />);
    const btn = await screen.findByRole("button", { name: /run dream pass/i });
    await userEvent.click(btn);

    await waitFor(() => {
      expect(
        screen.queryByText((c) =>
          c.includes("last dream") && c.includes("skipped (no_input)"),
        ),
      ).toBeDefined();
    });
  });
});

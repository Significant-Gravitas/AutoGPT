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

describe("MemoryVisualizer — admin graph canvas, rebuild + dream", () => {
  test("renders rebuild and dream buttons + the overview chip", async () => {
    setupHandlers();
    render(<MemoryVisualizer />);

    expect(
      await screen.findByRole("button", { name: /rebuild communities/i }),
    ).toBeDefined();
    expect(
      await screen.findByRole("button", { name: /run dream pass/i }),
    ).toBeDefined();
    await waitFor(() => {
      expect(
        screen.queryByText((c) => /12/.test(c) && /entit/i.test(c)),
      ).toBeDefined();
    });
  });

  test("clicking 'Rebuild communities' surfaces the last-rebuild chip", async () => {
    setupHandlers();
    render(<MemoryVisualizer />);

    const btn = await screen.findByRole("button", {
      name: /rebuild communities/i,
    });
    await userEvent.click(btn);

    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: /rebuilding…/i }),
      ).toBeDefined();
    });
    await waitFor(() => {
      expect(
        screen.queryByText((c) =>
          c.includes("last rebuild") && c.includes("1.2s"),
        ),
      ).toBeDefined();
    });
  });

  test("rebuild skip result surfaces a skip-reason chip", async () => {
    server.use(
      getGetV2GetMemoryOverviewMockHandler200(
        getGetV2GetMemoryOverviewResponseMock200(),
      ),
      getGetV2GetGraphMockHandler200(getGetV2GetGraphResponseMock200()),
      getPostV2RebuildCommunitiesMockHandler200({
        ...getPostV2RebuildCommunitiesResponseMock200(),
        skipped: true,
        skip_reason: "no_activity",
        elapsed_seconds: 0.05,
      }),
    );
    render(<MemoryVisualizer />);
    const btn = await screen.findByRole("button", {
      name: /rebuild communities/i,
    });
    await userEvent.click(btn);

    await waitFor(() => {
      expect(
        screen.queryByText((c) =>
          c.includes("last rebuild") &&
          c.includes("skipped (no_activity)"),
        ),
      ).toBeDefined();
    });
  });

  test("clicking 'Run dream pass' surfaces the last-dream chip with per-phase counts", async () => {
    setupHandlers();
    render(<MemoryVisualizer />);

    const btn = await screen.findByRole("button", { name: /run dream pass/i });
    await userEvent.click(btn);

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /dreaming…/i })).toBeDefined();
    });
    await waitFor(() => {
      expect(
        screen.queryByText(
          (c) =>
            c.includes("last dream") &&
            c.includes("w=3") &&
            c.includes("p=2") &&
            c.includes("d=1"),
        ),
      ).toBeDefined();
    });
  });

  test("dream skip result surfaces a skip-reason chip", async () => {
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

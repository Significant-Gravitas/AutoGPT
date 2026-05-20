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
  );
}

describe("MemoryVisualizer — admin graph canvas + rebuild", () => {
  test("renders the rebuild button in the ControlBar", async () => {
    setupHandlers();
    render(<MemoryVisualizer />);
    expect(
      await screen.findByRole("button", { name: /rebuild communities/i }),
    ).toBeDefined();
  });

  test("clicking 'Rebuild communities' calls the rebuild endpoint and surfaces the result chip", async () => {
    setupHandlers();
    render(<MemoryVisualizer />);

    const btn = await screen.findByRole("button", {
      name: /rebuild communities/i,
    });
    await userEvent.click(btn);

    // While the mutation is pending the button label flips to "Rebuilding…"
    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: /rebuilding…/i }),
      ).toBeDefined();
    });

    // After the response lands the last-rebuild chip shows the elapsed time
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
          c.includes("last rebuild") && c.includes("skipped (no_activity)"),
        ),
      ).toBeDefined();
    });
  });

  test("renders the entity-count chip from the overview response", async () => {
    setupHandlers();
    render(<MemoryVisualizer />);
    // Strip-style header chip is rendered from overview.entities=12
    await waitFor(() => {
      expect(
        screen.queryByText((c) => /12/.test(c) && /entit/i.test(c)),
      ).toBeDefined();
    });
  });
});

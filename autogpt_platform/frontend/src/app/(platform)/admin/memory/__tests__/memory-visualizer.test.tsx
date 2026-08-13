import { describe, expect, test, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";

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

    await screen.findByRole("button", { name: /rebuild communities/i });
    await screen.findByRole("button", { name: /dream pass/i });
    // Overview chip is rendered as two adjacent elements ("12" + "Entities"),
    // so assert each label individually.
    await screen.findByText("12");
    await screen.findByText(/^Entities$/i);
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
    // findByRole throws if the morphed label never appears, so this
    // genuinely fails on a broken poll loop.
    await screen.findByRole("button", { name: /rebuild…/i });
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

    await screen.findByRole("button", {
      name: /batch submitted \(consolidate\)/i,
    });
  });

  test("status-endpoint 500 stops polling + reactivates the dream button", async () => {
    setupBaseHandlers();
    server.use(
      getPostV2TriggerDreamPassMockHandler202({
        ...getPostV2TriggerDreamPassResponseMock202(),
        job_id: "job-dream-err",
        state: "queued",
      }),
      // Status endpoint flakes — the poll loop must stop and the
      // button must come back, not spin forever.
      http.get("*/api/admin/memory/:userId/dream/:jobId", () =>
        HttpResponse.json({ error: "boom" }, { status: 500 }),
      ),
    );
    render(<MemoryVisualizer />);
    const btn = await screen.findByRole("button", { name: /dream pass/i });
    await userEvent.click(btn);

    // The trigger button reactivates to its idle label after the
    // status endpoint fails — proves the active job id was cleared.
    await waitFor(
      async () => {
        const idle = await screen.findByRole("button", { name: /dream pass/i });
        // The label morphs to "Dreaming…" briefly while we're between
        // the POST and the error; check exact-match to the idle text.
        expect(idle.textContent?.trim()).toBe("Dream pass");
      },
      { timeout: 5_000 },
    );
  });

  test(
    "status poll failing AFTER a successful poll still reactivates the dream button",
    { timeout: 20_000 },
    async () => {
      setupBaseHandlers();
      let statusCalls = 0;
      server.use(
        getPostV2TriggerDreamPassMockHandler202({
          ...getPostV2TriggerDreamPassResponseMock202(),
          job_id: "job-dream-stale",
          state: "queued",
        }),
        // First poll succeeds (running), every later poll 500s. React
        // Query keeps the stale running status while flipping the query
        // into error state — the terminal handler must treat the error
        // as terminal even though stale data exists, or the button is
        // stuck on "Dreaming…" forever with polling stopped.
        http.get("*/api/admin/memory/:userId/dream/:jobId", () => {
          statusCalls += 1;
          if (statusCalls === 1) {
            return HttpResponse.json({
              ...getGetV2GetDreamPassStatusResponseMock200(),
              job_id: "job-dream-stale",
              kind: "dream_pass",
              state: "running",
              current_phase: "consolidate",
            });
          }
          return HttpResponse.json({ error: "boom" }, { status: 500 });
        }),
      );
      render(<MemoryVisualizer />);
      const btn = await screen.findByRole("button", { name: /dream pass/i });
      await userEvent.click(btn);

      // First poll lands → phase-aware label proves status data is cached.
      await screen.findByRole(
        "button",
        { name: /consolidate…/i },
        { timeout: 5_000 },
      );

      // Second poll (one POLL_INTERVAL_MS later) errors → active job id
      // must clear and the idle label must come back.
      await waitFor(
        async () => {
          const idle = await screen.findByRole("button", {
            name: /dream pass/i,
          });
          expect(idle.textContent?.trim()).toBe("Dream pass");
        },
        { timeout: 10_000 },
      );
      expect(statusCalls).toBeGreaterThanOrEqual(2);
    },
  );
});

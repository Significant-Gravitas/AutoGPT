import { describe, expect, test, vi } from "vitest";
import { useState } from "react";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { useQueryClient } from "@tanstack/react-query";

import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2GetExpertGraphQueryKey,
  getGetV2GetExpertMemoryOverviewQueryKey,
  getGetV2GetGraphQueryKey,
  getGetV2GetMemoryOverviewQueryKey,
} from "@/app/api/__generated__/endpoints/admin/admin";
import {
  getGetV2GetCommunityRebuildStatusMockHandler200,
  getGetV2GetCommunityRebuildStatusResponseMock200,
  getGetV2GetDreamPassStatusMockHandler200,
  getGetV2GetDreamPassStatusResponseMock200,
  getGetV2GetGraphResponseMock200,
  getGetV2GetMemoryOverviewResponseMock200,
  getPostV2RebuildCommunitiesMockHandler202,
  getPostV2RebuildCommunitiesResponseMock202,
  getPostV2TriggerDreamPassMockHandler202,
  getPostV2TriggerDreamPassResponseMock202,
} from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";

const { toastMock } = vi.hoisted(() => ({ toastMock: vi.fn() }));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: toastMock }),
}));

// react-force-graph-2d uses HTMLCanvas + window APIs and pulls in d3 at
// import time. In the jsdom test env neither exists, so swap it for a
// trivial stub so MemoryVisualizer's tree is renderable.
vi.mock("react-force-graph-2d", () => ({
  default: () => null,
}));

import { MemoryVisualizer } from "../components/MemoryVisualizer";
import { useMemoryVisualizer } from "../components/useMemoryVisualizer";

function makeExpert(id: string, name: string, role = "Researcher"): Expert {
  return {
    id,
    name,
    avatar_url: null,
    role,
    tagline: null,
    bio: null,
    skills: [],
    identity: "",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: null,
    is_archived: false,
    workflows: [],
  };
}

// Account and expert scopes are separate (path-scoped) endpoints. The
// generated response mocks randomize ``expert_id`` and the hook's scope
// tripwire refuses payloads whose echoed expert_id disagrees with the
// requested scope, so each handler pins the echo for its own scope.
function setupBaseHandlers(experts: Expert[] = []) {
  server.use(
    getListExpertsMockHandler200(experts),
    http.get("*/api/admin/memory/:userId/overview", () =>
      HttpResponse.json({
        ...getGetV2GetMemoryOverviewResponseMock200(),
        expert_id: null,
        user_id: "u-1",
        group_id: "g-1",
        entities: 12,
        episodes: 30,
        relates_to_edges: 25,
        mentions_edges: 30,
        communities: 0,
      }),
    ),
    http.get("*/api/admin/memory/:userId/graph", () =>
      HttpResponse.json({
        ...getGetV2GetGraphResponseMock200(),
        expert_id: null,
        user_id: "u-1",
        group_id: "g-1",
        nodes: [],
        edges: [],
        truncated: false,
      }),
    ),
    http.get(
      "*/api/admin/memory/:userId/experts/:expertId/overview",
      ({ params }) =>
        HttpResponse.json({
          ...getGetV2GetMemoryOverviewResponseMock200(),
          expert_id: String(params.expertId),
          user_id: "u-1",
          group_id: "g-expert",
          entities: 3,
          episodes: 4,
          relates_to_edges: 2,
          mentions_edges: 1,
          communities: 0,
        }),
    ),
    http.get(
      "*/api/admin/memory/:userId/experts/:expertId/graph",
      ({ params }) =>
        HttpResponse.json({
          ...getGetV2GetGraphResponseMock200(),
          expert_id: String(params.expertId),
          user_id: "u-1",
          group_id: "g-expert",
          nodes: [],
          edges: [],
          truncated: false,
        }),
    ),
  );
}

function MemoryActionHarness() {
  const memory = useMemoryVisualizer("expert-ada");
  return (
    <>
      <button type="button" onClick={memory.triggerRebuild}>
        Rebuild callback
      </button>
      <button type="button" onClick={memory.triggerDream}>
        Dream callback
      </button>
      <button type="button" onClick={memory.triggerRatification}>
        Ratification callback
      </button>
      <button type="button" onClick={memory.triggerNightly}>
        Nightly callback
      </button>
      <span data-testid="mutation-states">
        {[
          memory.rebuild.isIdle,
          memory.dream.isIdle,
          memory.ratification.isIdle,
          memory.nightly.isIdle,
        ].join(",")}
      </span>
    </>
  );
}

function MemoryVisualizerWithRefetch() {
  const queryClient = useQueryClient();
  return (
    <>
      <MemoryVisualizer />
      <button type="button" onClick={() => queryClient.invalidateQueries()}>
        Refetch experts
      </button>
    </>
  );
}

const PERSISTENT_GRAPH_PARAMS = {
  include_episodes: true,
  include_communities: true,
  node_limit: 10000,
  edge_limit: 20000,
};

function MemoryVisualizerWithPersistentGraphCache() {
  const queryClient = useQueryClient();
  const [ready, setReady] = useState(false);
  const [expertCacheInvalidated, setExpertCacheInvalidated] = useState<
    string | undefined
  >();

  function loadVisualizer() {
    queryClient.setQueryDefaults(getGetV2GetGraphQueryKey("me"), {
      staleTime: Infinity,
    });
    queryClient.setQueryDefaults(getGetV2GetMemoryOverviewQueryKey("me"), {
      staleTime: Infinity,
    });
    queryClient.setQueryDefaults(
      getGetV2GetExpertGraphQueryKey("me", "expert-ada"),
      { staleTime: Infinity },
    );
    queryClient.setQueryDefaults(
      getGetV2GetExpertMemoryOverviewQueryKey("me", "expert-ada"),
      { staleTime: Infinity },
    );
    setReady(true);
  }

  function inspectExpertCache() {
    setExpertCacheInvalidated(
      [
        queryClient.getQueryState(
          getGetV2GetExpertMemoryOverviewQueryKey("me", "expert-ada"),
        )?.isInvalidated,
        queryClient.getQueryState(
          getGetV2GetExpertGraphQueryKey(
            "me",
            "expert-ada",
            PERSISTENT_GRAPH_PARAMS,
          ),
        )?.isInvalidated,
      ].join(","),
    );
  }

  if (!ready) {
    return (
      <button type="button" onClick={loadVisualizer}>
        Load persistent graph cache
      </button>
    );
  }

  return (
    <>
      <MemoryVisualizer />
      <button type="button" onClick={inspectExpertCache}>
        Inspect expert cache
      </button>
      <output data-testid="expert-cache-invalidated">
        {String(expertCacheInvalidated)}
      </output>
    </>
  );
}

describe("MemoryVisualizer — memory scope", () => {
  test("uses the account memory endpoints by default", async () => {
    setupBaseHandlers();
    const overviewScopes: Array<string | null> = [];
    const graphScopes: Array<string | null> = [];
    server.use(
      http.get("*/api/admin/memory/:userId/overview", () => {
        overviewScopes.push(null);
        return HttpResponse.json({
          ...getGetV2GetMemoryOverviewResponseMock200(),
          expert_id: null,
          entities: 12,
        });
      }),
      http.get("*/api/admin/memory/:userId/graph", () => {
        graphScopes.push(null);
        return HttpResponse.json({
          ...getGetV2GetGraphResponseMock200(),
          expert_id: null,
          nodes: [],
          edges: [],
          truncated: false,
        });
      }),
      http.get(
        "*/api/admin/memory/:userId/experts/:expertId/overview",
        ({ params }) => {
          overviewScopes.push(String(params.expertId));
          return HttpResponse.json({
            ...getGetV2GetMemoryOverviewResponseMock200(),
            expert_id: String(params.expertId),
          });
        },
      ),
      http.get(
        "*/api/admin/memory/:userId/experts/:expertId/graph",
        ({ params }) => {
          graphScopes.push(String(params.expertId));
          return HttpResponse.json({
            ...getGetV2GetGraphResponseMock200(),
            expert_id: String(params.expertId),
            nodes: [],
            edges: [],
            truncated: false,
          });
        },
      ),
    );

    render(<MemoryVisualizer />);

    expect(
      (await screen.findByRole("combobox", { name: "Memory scope" }))
        .textContent,
    ).toContain("AutoPilot");
    await screen.findByText("12");
    await waitFor(() => {
      expect(overviewScopes).toEqual([null]);
      expect(graphScopes).toEqual([null]);
    });
    expect(screen.getByRole("button", { name: /dream pass/i })).toBeDefined();
  });

  test("selecting an expert scopes reads and hides memory maintenance", async () => {
    setupBaseHandlers([makeExpert("expert-ada", "Ada")]);
    const overviewScopes: Array<string | null> = [];
    const graphScopes: Array<string | null> = [];
    server.use(
      http.get(
        "*/api/admin/memory/:userId/experts/:expertId/overview",
        ({ params }) => {
          overviewScopes.push(String(params.expertId));
          return HttpResponse.json({
            ...getGetV2GetMemoryOverviewResponseMock200(),
            expert_id: String(params.expertId),
          });
        },
      ),
      http.get(
        "*/api/admin/memory/:userId/experts/:expertId/graph",
        ({ params }) => {
          graphScopes.push(String(params.expertId));
          return HttpResponse.json({
            ...getGetV2GetGraphResponseMock200(),
            expert_id: String(params.expertId),
            nodes: [],
            edges: [],
            truncated: false,
          });
        },
      ),
    );

    render(<MemoryVisualizer />);

    const selector = await screen.findByRole("combobox", {
      name: "Memory scope",
    });
    await waitFor(() => {
      expect((selector as HTMLButtonElement).disabled).toBe(false);
    });
    fireEvent.click(selector);
    fireEvent.click(
      await screen.findByRole("option", { name: /Ada — Researcher/ }),
    );

    expect(
      await screen.findByText("Expert memory is read-only."),
    ).toBeDefined();
    await waitFor(() => {
      expect(overviewScopes).toContain("expert-ada");
      expect(graphScopes).toContain("expert-ada");
    });
    expect(
      screen.queryByRole("button", { name: /rebuild communities/i }),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: /dream pass/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /ratification/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /nightly batch/i })).toBeNull();
    expect(screen.queryByTestId("dream-result-panel")).toBeNull();
    expect(
      screen.getByRole("checkbox", { name: /communities/i }),
    ).toBeDefined();
  });

  test("scope switches preserve in-flight AutoPilot job tracking", async () => {
    setupBaseHandlers([makeExpert("expert-ada", "Ada")]);
    let dreamRequests = 0;
    server.use(
      http.post("*/api/admin/memory/:userId/dream", () => {
        dreamRequests += 1;
        return HttpResponse.json(
          {
            ...getPostV2TriggerDreamPassResponseMock202(),
            job_id: "job-dream-switch",
            state: "queued",
          },
          { status: 202 },
        );
      }),
      getGetV2GetDreamPassStatusMockHandler200({
        ...getGetV2GetDreamPassStatusResponseMock200(),
        job_id: "job-dream-switch",
        kind: "dream_pass",
        state: "running",
        current_phase: "consolidate",
      }),
    );
    render(<MemoryVisualizer />);

    await userEvent.click(
      await screen.findByRole("button", { name: /dream pass/i }),
    );
    await screen.findByRole("button", { name: /consolidate…/i });

    const selector = screen.getByRole("combobox", { name: "Memory scope" });
    fireEvent.click(selector);
    fireEvent.click(
      await screen.findByRole("option", { name: /Ada — Researcher/ }),
    );
    await screen.findByText("Expert memory is read-only.");

    fireEvent.click(selector);
    fireEvent.click(await screen.findByRole("option", { name: /AutoPilot/i }));

    const activeButton = await screen.findByRole("button", {
      name: /consolidate…/i,
    });
    expect((activeButton as HTMLButtonElement).disabled).toBe(true);
    expect(dreamRequests).toBe(1);
  });

  test("expert callbacks stay read-only even when invoked directly", async () => {
    setupBaseHandlers([makeExpert("expert-ada", "Ada")]);
    render(<MemoryActionHarness />);

    await userEvent.click(
      screen.getByRole("button", { name: "Rebuild callback" }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Dream callback" }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Ratification callback" }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Nightly callback" }),
    );

    expect(screen.getByTestId("mutation-states").textContent).toBe(
      "true,true,true,true",
    );
  });

  test("expert-list errors keep AutoPilot memory available", async () => {
    setupBaseHandlers();
    server.use(
      http.get("*/api/experts", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<MemoryVisualizer />);

    await screen.findByText("Failed to load experts.");
    const selector = screen.getByRole("combobox", { name: "Memory scope" });
    expect((selector as HTMLButtonElement).disabled).toBe(false);
    expect(selector.textContent).toContain("AutoPilot");
    await screen.findByText("12");
  });

  test("removed experts reset the selected scope to AutoPilot", async () => {
    let experts = [makeExpert("expert-ada", "Ada")];
    setupBaseHandlers();
    server.use(http.get("*/api/experts", () => HttpResponse.json(experts)));
    render(<MemoryVisualizerWithRefetch />);

    const selector = await screen.findByRole("combobox", {
      name: "Memory scope",
    });
    await waitFor(() =>
      expect((selector as HTMLButtonElement).disabled).toBe(false),
    );
    fireEvent.click(selector);
    fireEvent.click(
      await screen.findByRole("option", { name: /Ada — Researcher/ }),
    );
    await screen.findByText("Expert memory is read-only.");

    experts = [];
    await userEvent.click(
      screen.getByRole("button", { name: "Refetch experts" }),
    );

    await waitFor(() => expect(selector.textContent).toContain("AutoPilot"));
    expect(screen.queryByText("Expert memory is read-only.")).toBeNull();
    expect(toastMock).toHaveBeenCalledWith({
      title: "Expert no longer available",
      description: "Showing AutoPilot account memory instead.",
    });
  });

  test("expert-list refetch errors keep the selected cached expert", async () => {
    let expertsRequestFails = false;
    setupBaseHandlers();
    server.use(
      http.get("*/api/experts", () =>
        expertsRequestFails
          ? HttpResponse.json({ detail: "boom" }, { status: 500 })
          : HttpResponse.json([makeExpert("expert-ada", "Ada")]),
      ),
    );
    render(<MemoryVisualizerWithRefetch />);

    const selector = await screen.findByRole("combobox", {
      name: "Memory scope",
    });
    await waitFor(() =>
      expect((selector as HTMLButtonElement).disabled).toBe(false),
    );
    fireEvent.click(selector);
    fireEvent.click(
      await screen.findByRole("option", { name: /Ada — Researcher/ }),
    );
    await screen.findByText("Expert memory is read-only.");
    toastMock.mockClear();

    expertsRequestFails = true;
    await userEvent.click(
      screen.getByRole("button", { name: "Refetch experts" }),
    );

    await screen.findByText("Failed to load experts.");
    expect(selector.textContent).toContain("Ada");
    expect(screen.getByText("Expert memory is read-only.")).toBeDefined();
    expect(toastMock).not.toHaveBeenCalled();
  });

  test("empty and error scope states are announced without blocking AutoPilot", async () => {
    setupBaseHandlers();
    render(<MemoryVisualizer />);

    const emptyState = await screen.findByText("No experts for this account.");
    expect(emptyState.closest('[aria-live="polite"]')).not.toBeNull();
    expect(
      (
        screen.getByRole("combobox", {
          name: "Memory scope",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
  });

  test("announces expert loading separately from an empty roster", async () => {
    setupBaseHandlers();
    server.use(
      http.get("*/api/experts", async () => {
        await delay(100);
        return HttpResponse.json([]);
      }),
    );

    render(<MemoryVisualizer />);

    expect(screen.getByText("Loading experts…")).toBeDefined();
    await screen.findByText("No experts for this account.");
  });

  test("only appends IDs when expert name and role labels collide", async () => {
    setupBaseHandlers([
      makeExpert("ada-one-1234", "Ada", "Researcher"),
      makeExpert("ada-two-5678", "Ada", "Writer"),
      makeExpert("ada-three-9012", "Ada", "Researcher"),
    ]);

    render(<MemoryVisualizer />);

    const selector = await screen.findByRole("combobox", {
      name: "Memory scope",
    });
    await waitFor(() =>
      expect((selector as HTMLButtonElement).disabled).toBe(false),
    );
    fireEvent.click(selector);

    expect(
      await screen.findByRole("option", {
        name: "Ada — Researcher (ada-one-1234)",
      }),
    ).toBeDefined();
    expect(
      screen.getByRole("option", { name: "Ada — Researcher (ada-three-9012)" }),
    ).toBeDefined();
    expect(screen.getByRole("option", { name: "Ada — Writer" })).toBeDefined();
  });

  test("switching scope swaps the rendered graph content, not just the request", async () => {
    setupBaseHandlers([makeExpert("expert-ada", "Ada")]);
    server.use(
      http.get("*/api/admin/memory/:userId/graph", () =>
        HttpResponse.json({
          ...getGetV2GetGraphResponseMock200(),
          expert_id: null,
          nodes: [
            {
              uuid: "n-auto-1",
              label: "Entity",
              type: "AutoPilotFact",
              name: "Account node",
            },
          ],
          edges: [],
          truncated: false,
        }),
      ),
      http.get(
        "*/api/admin/memory/:userId/experts/:expertId/graph",
        ({ params }) =>
          HttpResponse.json({
            ...getGetV2GetGraphResponseMock200(),
            expert_id: String(params.expertId),
            nodes: [
              {
                uuid: "n-expert-1",
                label: "Entity",
                type: "ExpertInsight",
                name: "Expert node",
              },
            ],
            edges: [],
            truncated: false,
          }),
      ),
    );

    render(<MemoryVisualizer />);

    // Account scope renders the account graph's node-type pill.
    await screen.findByRole("button", { name: "AutoPilotFact (1)" });

    const selector = screen.getByRole("combobox", { name: "Memory scope" });
    await waitFor(() =>
      expect((selector as HTMLButtonElement).disabled).toBe(false),
    );
    fireEvent.click(selector);
    fireEvent.click(
      await screen.findByRole("option", { name: /Ada — Researcher/ }),
    );

    // Expert scope renders the expert graph's pill and drops the
    // account one — proves the displayed data actually swapped.
    await screen.findByRole("button", { name: "ExpertInsight (1)" });
    expect(
      screen.queryByRole("button", { name: "AutoPilotFact (1)" }),
    ).toBeNull();

    fireEvent.click(selector);
    fireEvent.click(await screen.findByRole("option", { name: /AutoPilot/i }));

    await screen.findByRole("button", { name: "AutoPilotFact (1)" });
    expect(
      screen.queryByRole("button", { name: "ExpertInsight (1)" }),
    ).toBeNull();
  });

  test("account job status text hides while an expert scope is selected", async () => {
    setupBaseHandlers([makeExpert("expert-ada", "Ada")]);
    server.use(
      getPostV2TriggerDreamPassMockHandler202({
        ...getPostV2TriggerDreamPassResponseMock202(),
        job_id: "job-dream-hidden",
        state: "queued",
      }),
      getGetV2GetDreamPassStatusMockHandler200({
        ...getGetV2GetDreamPassStatusResponseMock200(),
        job_id: "job-dream-hidden",
        kind: "dream_pass",
        state: "running",
        current_phase: "consolidate",
      }),
    );
    render(<MemoryVisualizer />);

    await userEvent.click(
      await screen.findByRole("button", { name: /dream pass/i }),
    );
    await screen.findByText("dream: running (consolidate)");

    const selector = screen.getByRole("combobox", { name: "Memory scope" });
    fireEvent.click(selector);
    fireEvent.click(
      await screen.findByRole("option", { name: /Ada — Researcher/ }),
    );
    await screen.findByText("Expert memory is read-only.");

    // The account job keeps polling, but its status must not bleed
    // into the expert-scoped (read-only) control bar.
    expect(screen.queryByText("dream: running (consolidate)")).toBeNull();

    fireEvent.click(selector);
    fireEvent.click(await screen.findByRole("option", { name: /AutoPilot/i }));

    await screen.findByText("dream: running (consolidate)");
  });

  test("responses scoped to a different expert are rejected, not rendered", async () => {
    setupBaseHandlers([makeExpert("expert-ada", "Ada")]);
    toastMock.mockClear();
    server.use(
      // Malfunctioning server: the expert endpoints return account-
      // scoped payloads (expert_id echo of null) instead of the
      // requested expert's data.
      http.get("*/api/admin/memory/:userId/experts/:expertId/overview", () =>
        HttpResponse.json({
          ...getGetV2GetMemoryOverviewResponseMock200(),
          expert_id: null,
          entities: 999,
          episodes: 1,
          relates_to_edges: 1,
          mentions_edges: 1,
          communities: 1,
        }),
      ),
      http.get("*/api/admin/memory/:userId/experts/:expertId/graph", () =>
        HttpResponse.json({
          ...getGetV2GetGraphResponseMock200(),
          expert_id: null,
          nodes: [],
          edges: [],
          truncated: false,
        }),
      ),
    );

    render(<MemoryVisualizer />);

    // Account scope renders normally from the base handlers.
    await screen.findByText("12");

    const selector = screen.getByRole("combobox", { name: "Memory scope" });
    await waitFor(() =>
      expect((selector as HTMLButtonElement).disabled).toBe(false),
    );
    fireEvent.click(selector);
    fireEvent.click(
      await screen.findByRole("option", { name: /Ada — Researcher/ }),
    );

    // Expert scope now receives account-scoped payloads → tripwire.
    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Memory scope mismatch",
          variant: "destructive",
        }),
      );
    });
    expect(screen.queryByText("999")).toBeNull();
    expect(screen.getAllByText(/Memory scope mismatch/).length).toBeGreaterThan(
      0,
    );
  });
});

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

  test("job completion refreshes every AutoPilot cache without invalidating expert memory", async () => {
    setupBaseHandlers([makeExpert("expert-ada", "Ada")]);
    const graphRequests = new Map<string, number>();
    server.use(
      http.get("*/api/admin/memory/:userId/graph", ({ request }) => {
        const url = new URL(request.url);
        const requestKey = [
          "autopilot",
          url.searchParams.get("include_episodes"),
          url.searchParams.get("include_communities"),
        ].join(":");
        graphRequests.set(requestKey, (graphRequests.get(requestKey) ?? 0) + 1);
        return HttpResponse.json({
          ...getGetV2GetGraphResponseMock200(),
          expert_id: null,
          nodes: [],
          edges: [],
          truncated: false,
        });
      }),
      http.get(
        "*/api/admin/memory/:userId/experts/:expertId/graph",
        ({ request, params }) => {
          const url = new URL(request.url);
          const requestKey = [
            String(params.expertId),
            url.searchParams.get("include_episodes"),
            url.searchParams.get("include_communities"),
          ].join(":");
          graphRequests.set(
            requestKey,
            (graphRequests.get(requestKey) ?? 0) + 1,
          );
          return HttpResponse.json({
            ...getGetV2GetGraphResponseMock200(),
            expert_id: String(params.expertId),
            nodes: [],
            edges: [],
            truncated: false,
          });
        },
      ),
      getPostV2TriggerDreamPassMockHandler202({
        ...getPostV2TriggerDreamPassResponseMock202(),
        job_id: "job-dream-cache",
        state: "queued",
      }),
      getGetV2GetDreamPassStatusMockHandler200({
        ...getGetV2GetDreamPassStatusResponseMock200(),
        job_id: "job-dream-cache",
        kind: "dream_pass",
        state: "complete",
      }),
    );
    render(<MemoryVisualizerWithPersistentGraphCache />);
    await userEvent.click(
      screen.getByRole("button", { name: "Load persistent graph cache" }),
    );

    await waitFor(() =>
      expect(graphRequests.get("autopilot:false:true")).toBe(1),
    );
    const episodes = await screen.findByRole("checkbox", {
      name: /episodes/i,
    });
    await userEvent.click(episodes);
    await waitFor(() =>
      expect(graphRequests.get("autopilot:true:true")).toBe(1),
    );

    const selector = screen.getByRole("combobox", { name: "Memory scope" });
    await waitFor(() =>
      expect((selector as HTMLButtonElement).disabled).toBe(false),
    );
    fireEvent.click(selector);
    fireEvent.click(
      await screen.findByRole("option", { name: /Ada — Researcher/ }),
    );
    await waitFor(() =>
      expect(graphRequests.get("expert-ada:true:true")).toBe(1),
    );

    fireEvent.click(selector);
    fireEvent.click(await screen.findByRole("option", { name: /AutoPilot/i }));
    await userEvent.click(episodes);
    await userEvent.click(
      await screen.findByRole("button", { name: /dream pass/i }),
    );

    await waitFor(() =>
      expect(graphRequests.get("autopilot:false:true")).toBe(2),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Inspect expert cache" }),
    );
    expect(screen.getByTestId("expert-cache-invalidated").textContent).toBe(
      "false,false",
    );

    await userEvent.click(episodes);
    await waitFor(() =>
      expect(graphRequests.get("autopilot:true:true")).toBe(2),
    );
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

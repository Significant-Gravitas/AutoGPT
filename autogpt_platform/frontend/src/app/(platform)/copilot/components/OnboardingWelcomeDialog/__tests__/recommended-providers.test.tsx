import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

// Recommendations only exist behind the brain-dump flag; these tests are
// about what the panel does once it is on.
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === actual.Flag.ONBOARDING_BRAIN_DUMP,
  };
});

import { ConnectToolsPanel } from "../ConnectToolsPanel";

const PROVIDERS_URL =
  "http://localhost:3000/api/proxy/api/integrations/providers";
const CREDENTIALS_URL =
  "http://localhost:3000/api/proxy/api/integrations/credentials";
const RECOMMENDED_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/recommended-providers";

const POLL_INTERVAL_MS = 2_500;
const RECOMMENDED_HEADING = "Recommended from our conversation";
const SEARCH_PROMPT = "Search to find a service to connect.";

type RecommendedResponse = {
  ready: boolean;
  providers?: { provider: string; reason?: string }[] | null;
};

function stubRegistry() {
  server.use(
    http.get(PROVIDERS_URL, () =>
      HttpResponse.json([
        {
          name: "notion",
          description: "Docs and wikis",
          supported_auth_types: ["oauth2"],
        },
        {
          name: "slack",
          description: "Team chat",
          supported_auth_types: ["oauth2"],
        },
      ]),
    ),
    http.get(CREDENTIALS_URL, () => HttpResponse.json([])),
  );
}

// Serves one scripted response per poll, reusing the last one forever, and
// reports how many times the endpoint was actually hit.
function scriptRecommendations(...responses: RecommendedResponse[]) {
  const hits: number[] = [];
  server.use(
    http.get(RECOMMENDED_URL, () => {
      const index = Math.min(hits.length, responses.length - 1);
      hits.push(index);
      return HttpResponse.json(responses[index]);
    }),
  );
  return hits;
}

function renderPanel() {
  return render(<ConnectToolsPanel onBack={vi.fn()} onNext={vi.fn()} />);
}

async function pollTimes(count: number) {
  for (let i = 0; i < count; i++) {
    await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
  }
}

afterEach(() => {
  vi.useRealTimers();
});

describe("ConnectToolsPanel — recommendations", () => {
  it("renders the model's picks under the recommended heading", async () => {
    stubRegistry();
    scriptRecommendations({
      ready: true,
      providers: [
        { provider: "notion", reason: "You mentioned meeting notes" },
      ],
    });

    renderPanel();

    expect(await screen.findByText(RECOMMENDED_HEADING)).toBeDefined();
    expect(screen.getByRole("button", { name: /Notion/ })).toBeDefined();
    // The reason is the whole point of the section — it has to be on screen,
    // not just in the mapped data.
    expect(screen.getByText("You mentioned meeting notes")).toBeDefined();
    // Only what the model picked — not the whole registry.
    expect(screen.queryByRole("button", { name: /Slack/ })).toBeNull();
    expect(screen.queryByText(SEARCH_PROMPT)).toBeNull();
  });

  it("falls back to the registry blurb when the model gave no reason", async () => {
    stubRegistry();
    scriptRecommendations({
      ready: true,
      providers: [{ provider: "notion", reason: "" }],
    });

    renderPanel();

    expect(await screen.findByText("Docs and wikis")).toBeDefined();
  });

  it("drops recommendations whose provider is no longer in the registry", async () => {
    stubRegistry();
    scriptRecommendations({
      ready: true,
      providers: [
        { provider: "retired_tool", reason: "Renamed since the job ran" },
        { provider: "slack", reason: "You said your team lives in chat" },
      ],
    });

    renderPanel();

    expect(await screen.findByRole("button", { name: /Slack/ })).toBeDefined();
    expect(screen.queryByRole("button", { name: /Retired/ })).toBeNull();
    expect(screen.getAllByText(RECOMMENDED_HEADING)).toHaveLength(1);
  });

  it("shows the search prompt and stops polling when the model recommends nothing", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    stubRegistry();
    // An empty array is a real answer, not "still working".
    const hits = scriptRecommendations({ ready: true, providers: [] });

    renderPanel();

    await waitFor(() => expect(screen.getByText(SEARCH_PROMPT)).toBeDefined());
    expect(screen.queryByText(RECOMMENDED_HEADING)).toBeNull();

    await pollTimes(4);
    expect(hits).toHaveLength(1);
  });

  it("keeps polling while the job is unfinished and renders the answer when it lands", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    stubRegistry();
    // `providers: null` while `ready` is false is the "job still running"
    // signal — it must not be mistaken for "nothing to recommend".
    const hits = scriptRecommendations(
      { ready: false, providers: null },
      { ready: false, providers: null },
      {
        ready: true,
        providers: [{ provider: "notion", reason: "You mentioned notes" }],
      },
    );

    renderPanel();

    await waitFor(() => expect(screen.getByText(SEARCH_PROMPT)).toBeDefined());
    expect(hits).toHaveLength(1);

    await pollTimes(2);

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Notion/ })).toBeDefined(),
    );
    expect(screen.queryByText(SEARCH_PROMPT)).toBeNull();
    expect(hits).toHaveLength(3);

    // Results arrived, so the panel must go quiet.
    await pollTimes(4);
    expect(hits).toHaveLength(3);
  });

  it("gives up polling after roughly a minute when the job never reports ready", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    stubRegistry();
    const hits = scriptRecommendations({ ready: false, providers: null });

    renderPanel();

    await waitFor(() => expect(hits.length).toBeGreaterThan(0));

    await pollTimes(40);
    const settled = hits.length;
    // 24 refetches after the initial load — see MAX_POLLS in the hook.
    expect(settled).toBe(25);

    await pollTimes(10);
    expect(hits).toHaveLength(settled);
  });
});

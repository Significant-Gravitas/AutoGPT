import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { onlineManager, QueryClient } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";

import { getGetV1GetSharedExecutionMockHandler200 } from "@/app/api/__generated__/endpoints/default/default.msw";
import type { SharedExecutionResponse } from "@/app/api/__generated__/models/sharedExecutionResponse";
import { resetClientRequestFailureLog } from "@/app/api/mutators/request-failure-log";
import { server } from "@/mocks/mock-server";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import SharePage from "../page";

const TOKEN = "550e8400-e29b-41d4-a716-446655440000";

const mockUseParams = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useParams: mockUseParams,
  usePathname: () => `/share/${TOKEN}`,
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
}));

// RunOutputs pulls in heavy library code (output renderers per
// block type); stub it to keep this page test focused on the chrome
// (the wrapper layout + alert + summary card) we actually touched.
vi.mock(
  "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/SelectedRunView/components/RunOutputs",
  () => ({
    RunOutputs: () => <div data-testid="run-outputs-stub" />,
  }),
);

beforeEach(() => {
  vi.clearAllMocks();
  mockUseParams.mockReturnValue({ token: TOKEN });
});

afterEach(() => {
  resetClientRequestFailureLog();
});

function weatherRun(): SharedExecutionResponse {
  return {
    id: "exec-1",
    graph_name: "Weather Agent",
    graph_description: "Reports the weather",
    status: "COMPLETED",
    created_at: new Date("2026-05-12T00:00:00Z"),
    outputs: {},
  };
}

// The app's client keeps React Query's three retries; this one only drops the
// backoff between them, so a test can count them without waiting ~7 s.
function createRetryingQueryClient() {
  return new QueryClient({ defaultOptions: { queries: { retryDelay: 0 } } });
}

// Answers every read of a shared run with `respond`, and returns the tokens
// the page asked for.
function answerSharedRunWith(respond: () => Response) {
  const tokens: string[] = [];
  server.use(
    http.get("*/api/public/shared/:token", ({ params }) => {
      tokens.push(String(params.token));
      return respond();
    }),
  );
  return tokens;
}

describe("SharePage (execution share viewer)", () => {
  test("renders the AutoGPT logo header + run summary + outputs on the happy path", async () => {
    server.use(getGetV1GetSharedExecutionMockHandler200(weatherRun));

    render(<SharePage />);

    // Logo (rendered inside the chrome wrapper at the top of the page).
    // Both light + dark variants ship in the DOM behind dark: class
    // selectors, so we look for at least one match.
    const logos = await screen.findAllByAltText("AutoGPT");
    expect(logos.length).toBeGreaterThan(0);
    // Public-share affordance alert.
    expect(
      await screen.findByText(/publicly shared workflow run result/i),
    ).toBeDefined();
    // Run name surfaces from the structured payload (in both header
    // and card-title slots; we just need it to be present).
    const titles = await screen.findAllByText("Weather Agent");
    expect(titles.length).toBeGreaterThan(0);
    // RunOutputs stub is mounted.
    expect(await screen.findByTestId("run-outputs-stub")).toBeDefined();
  });

  test.each([
    {
      link: "a revoked link (404)",
      token: TOKEN,
      status: 404,
      detail: "Shared execution not found",
    },
    {
      link: "a truncated link (422)",
      token: TOKEN.slice(0, -4),
      status: 422,
      detail: [
        {
          type: "string_pattern_mismatch",
          loc: ["path", "share_token"],
          msg: "String should match pattern",
        },
      ],
    },
  ])(
    "says 'Share Link Not Found' for $link at once, asking the backend only once and offering no retry",
    async ({ token, status, detail }) => {
      mockUseParams.mockReturnValue({ token });
      const requested = answerSharedRunWith(() =>
        HttpResponse.json({ detail }, { status }),
      );

      render(<SharePage />, { queryClient: createRetryingQueryClient() });

      expect(
        await screen.findByRole("heading", { name: "Share Link Not Found" }),
      ).toBeDefined();
      expect(screen.getByText(/invalid or has been disabled/i)).toBeDefined();
      expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
      expect(requested).toEqual([token]);
    },
  );

  test.each([
    {
      failure: "a 500 from the backend",
      respond: () =>
        HttpResponse.json({ detail: "Internal Server Error" }, { status: 500 }),
    },
    {
      failure: "a 502 from the proxy (backend down)",
      respond: () =>
        HttpResponse.json({ error: "Proxy request failed" }, { status: 502 }),
    },
    {
      failure: "a network error",
      respond: () => HttpResponse.error(),
    },
  ])(
    "says 'Unable to Load', not 'Share Link Not Found', after retrying $failure, and offers 'Try again'",
    async ({ respond }) => {
      const requested = answerSharedRunWith(respond);

      render(<SharePage />, { queryClient: createRetryingQueryClient() });

      expect(
        await screen.findByRole("heading", { name: "Unable to Load" }),
      ).toBeDefined();
      expect(screen.getByText(/try refreshing the page/i)).toBeDefined();
      expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
      expect(
        screen.queryByRole("heading", { name: "Share Link Not Found" }),
      ).toBeNull();
      // The first try and React Query's three retries.
      expect(requested).toHaveLength(4);
    },
  );

  test("says 'Share Link Not Found' for a 200 that carries no run, without retrying", async () => {
    mockUseParams.mockReturnValue({ token: TOKEN });
    const requested = answerSharedRunWith(() =>
      HttpResponse.json(null, { status: 200 }),
    );

    render(<SharePage />, { queryClient: createRetryingQueryClient() });

    expect(
      await screen.findByRole("heading", { name: "Share Link Not Found" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
    expect(requested).toEqual([TOKEN]);
  });

  test("says 'Share Link Not Found' when the link is revoked after the run has loaded", async () => {
    let revoked = false;
    answerSharedRunWith(() =>
      revoked
        ? HttpResponse.json({ detail: "Not found" }, { status: 404 })
        : HttpResponse.json(weatherRun()),
    );
    const queryClient = createRetryingQueryClient();
    render(<SharePage />, { queryClient });
    expect(
      (await screen.findAllByText("Weather Agent")).length,
    ).toBeGreaterThan(0);

    revoked = true;
    await act(() => queryClient.refetchQueries());

    expect(
      await screen.findByRole("heading", { name: "Share Link Not Found" }),
    ).toBeDefined();
  });

  test("keeps loading while the visitor is offline, then shows the run", async () => {
    server.use(getGetV1GetSharedExecutionMockHandler200(weatherRun));
    onlineManager.setOnline(false);
    try {
      render(<SharePage />);

      expect(
        await screen.findByText("Loading shared execution..."),
      ).toBeDefined();
      expect(
        screen.queryByRole("heading", { name: "Share Link Not Found" }),
      ).toBeNull();

      onlineManager.setOnline(true);

      const titles = await screen.findAllByText("Weather Agent");
      expect(titles.length).toBeGreaterThan(0);
    } finally {
      onlineManager.setOnline(true);
    }
  });

  describe("Cookie settings", () => {
    afterEach(() => {
      removeCookiebot();
      vi.unstubAllEnvs();
    });

    test("opens the Cookiebot dialog from the page footer", async () => {
      configureCookiebot();
      const { renew } = installCookiebot();
      server.use(
        http.get(
          "*/api/public/shared/:token",
          () => new HttpResponse(null, { status: 404 }),
        ),
      );

      render(<SharePage />);

      fireEvent.click(
        await screen.findByRole("button", { name: "Cookie settings" }),
      );

      expect(renew).toHaveBeenCalledOnce();
    });

    test("hides the link when no consent banner is configured", async () => {
      vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
      server.use(
        http.get(
          "*/api/public/shared/:token",
          () => new HttpResponse(null, { status: 404 }),
        ),
      );

      render(<SharePage />);

      expect(await screen.findByText(/share link not found/i)).toBeDefined();
      expect(
        screen.queryByRole("button", { name: "Cookie settings" }),
      ).toBeNull();
    });
  });
});

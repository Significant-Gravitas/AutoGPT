import { beforeEach, describe, expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";

import { getGetV1GetSharedExecutionMockHandler200 } from "@/app/api/__generated__/endpoints/default/default.msw";
import type { SharedExecutionResponse } from "@/app/api/__generated__/models/sharedExecutionResponse";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import SharePage from "../page";

const TOKEN = "550e8400-e29b-41d4-a716-446655440000";

vi.mock("next/navigation", () => ({
  useParams: () => ({ token: TOKEN }),
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
});

describe("SharePage (execution share viewer)", () => {
  test("renders the AutoGPT logo header + run summary + outputs on the happy path", async () => {
    server.use(
      getGetV1GetSharedExecutionMockHandler200(
        (): SharedExecutionResponse => ({
          id: "exec-1",
          graph_name: "Weather Agent",
          graph_description: "Reports the weather",
          status: "COMPLETED",
          created_at: new Date("2026-05-12T00:00:00Z"),
          outputs: {},
        }),
      ),
    );

    render(<SharePage />);

    // Logo (rendered inside the chrome wrapper at the top of the page).
    // Both light + dark variants ship in the DOM behind dark: class
    // selectors, so we look for at least one match.
    const logos = await screen.findAllByAltText("AutoGPT");
    expect(logos.length).toBeGreaterThan(0);
    // Public-share affordance alert.
    expect(
      await screen.findByText(/publicly shared agent run result/i),
    ).toBeDefined();
    // Run name surfaces from the structured payload (in both header
    // and card-title slots; we just need it to be present).
    const titles = await screen.findAllByText("Weather Agent");
    expect(titles.length).toBeGreaterThan(0);
    // RunOutputs stub is mounted.
    expect(await screen.findByTestId("run-outputs-stub")).toBeDefined();
  });

  test("renders the 'Share Link Not Found' card on a 404 from the backend", async () => {
    server.use(
      http.get(
        "*/api/public/shared/:token",
        () => new HttpResponse(null, { status: 404 }),
      ),
    );

    render(<SharePage />);

    expect(await screen.findByText(/share link not found/i)).toBeDefined();
    expect(
      await screen.findByText(/invalid or has been disabled/i),
    ).toBeDefined();
    // Try again button is the only retry affordance on this view.
    expect(screen.getByRole("button", { name: /try again/i })).toBeDefined();
  });
});

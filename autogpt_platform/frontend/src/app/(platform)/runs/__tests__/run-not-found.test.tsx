import { QueryClient } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";

import RunPage from "../[graphId]/[runId]/page";

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useParams: () => ({ graphId: "bad", runId: "bad" }),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/runs/bad/bad",
  useSearchParams: () => new URLSearchParams(),
}));

/** The app's own retry defaults, which the test client turns off: a run
 *  that cannot be read must not wait out three backed-off retries. */
function appQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { staleTime: 60_000 } },
  });
}

describe("Run page for a run that cannot be read", () => {
  it.each([
    [404, { detail: "Graph execution #bad not found." }],
    [422, { detail: [{ msg: "Input should be a valid UUID" }] }],
    [500, { detail: "Internal Server Error" }],
  ])(
    "ends in a not-found state with a way back after a %i",
    async (status, body) => {
      server.use(
        http.get("/api/proxy/api/graphs/bad/executions/bad", () =>
          HttpResponse.json(body, { status }),
        ),
        http.get("/api/proxy/api/library/agents/by-graph/bad", () =>
          HttpResponse.json({ detail: "Not found" }, { status: 404 }),
        ),
      );
      render(<RunPage />, { queryClient: appQueryClient() });

      expect(
        await screen.findByText("This run could not be found", undefined, {
          timeout: 2500,
        }),
      ).toBeDefined();
      expect(
        screen.getByRole("link", { name: "Open Schedules" }),
      ).toBeDefined();
      expect(screen.getByRole("button", { name: "Go back" })).toBeDefined();
    },
  );
});

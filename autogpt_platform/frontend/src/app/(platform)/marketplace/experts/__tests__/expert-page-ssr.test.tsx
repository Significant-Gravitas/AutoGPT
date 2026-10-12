import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactElement } from "react";
import { renderToString } from "react-dom/server";
import { beforeEach, describe, expect, test, vi } from "vitest";
import MarketplaceExpertPage from "../[expertId]/page";

const mockListExpertTemplates = vi.hoisted(() => vi.fn());
const mockParams = vi.hoisted(() => ({ expertId: "template-maria" }));

vi.mock(
  "@/app/api/__generated__/endpoints/experts/experts",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("@/app/api/__generated__/endpoints/experts/experts")
    >()),
    listExpertTemplates: mockListExpertTemplates,
  }),
);

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => `/marketplace/experts/${mockParams.expertId}`,
  useSearchParams: () => new URLSearchParams(),
  useParams: () => mockParams,
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

// The session only resolves in the browser, so on the server it is loading.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isLoggedIn: false, isUserLoading: true }),
}));

const maria = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  job_title: "Marketing Strategist",
  bio: "Maria is a senior marketing strategist with fifteen years across B2B SaaS.",
  skills: [],
  bundled_skills: [
    {
      id: "listing-1",
      slug: "brand-voice-guide",
      title: "Brand voice guide",
      description: "Keeps every draft on-brand.",
    },
  ],
  day_one: [
    {
      title: "Social listening on your brand",
      description:
        "Tracks mentions of your brand across X, LinkedIn and Reddit.",
      timing: "first scan · 1 hr",
    },
  ],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  weekly_budget: 500,
  workflows: [
    {
      id: "wf-1",
      name: "LinkedIn Post Generator",
      description: "Create research-driven LinkedIn posts in minutes.",
      store_listing_version_id: null,
      library_agent_id: null,
      graph_id: null,
      integration_providers: ["anthropic"],
    },
  ],
};

function okResponse(data: unknown) {
  return { status: 200, data, headers: new Headers() };
}

// What Next does with the page's output: one pass to HTML, no effects, so
// nothing the client would fetch after mount can leak into the result.
function renderServerHTML(page: ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return renderToString(
    <QueryClientProvider client={queryClient}>
      <NuqsTestingAdapter>
        <BackendAPIProvider>
          <TooltipProvider>{page}</TooltipProvider>
        </BackendAPIProvider>
      </NuqsTestingAdapter>
    </QueryClientProvider>,
  );
}

function renderPage(expertId: string) {
  mockParams.expertId = expertId;
  return MarketplaceExpertPage({ params: Promise.resolve({ expertId }) });
}

describe("Marketplace expert page on the server", () => {
  const requests: string[] = [];

  beforeEach(() => {
    mockListExpertTemplates.mockReset();
    // Shared across tests in the browser build of the query client.
    getQueryClient().clear();
    requests.length = 0;
    server.events.removeAllListeners();
    server.events.on("request:start", ({ request }) => {
      requests.push(request.url);
    });
  });

  test("puts the expert's text in the HTML without a client fetch", async () => {
    mockListExpertTemplates.mockResolvedValue(okResponse([maria]));

    const html = renderServerHTML(await renderPage("template-maria"));

    expect(html).toMatch(/<h1[^>]*>[^<]*Maria/);
    expect(html).toContain("Marketing Strategist");
    expect(html).toContain("Grows your brand while you sleep");
    expect(html).toContain(
      "Maria is a senior marketing strategist with fifteen years across B2B SaaS.",
    );
    expect(html).toContain("Social listening on your brand");
    expect(html).toContain("Brand voice guide");
    expect(html).toContain("LinkedIn Post Generator");
    expect(mockListExpertTemplates).toHaveBeenCalledTimes(1);
    expect(requests).toEqual([]);
  });

  test("shows no skeleton in place of the profile", async () => {
    mockListExpertTemplates.mockResolvedValue(okResponse([maria]));

    const html = renderServerHTML(await renderPage("template-maria"));

    // Only the header's hire action waits for the session and the flag; the
    // profile that follows the header must be text, not placeholders.
    const [header, body] = html.split("</header>");
    expect(header).toContain("<h1");
    expect(body).toContain("Maria is a senior marketing strategist");
    expect(body).not.toContain("animate-pulse");
  });

  test("404s on the server for an id that matches no template", async () => {
    mockListExpertTemplates.mockResolvedValue(okResponse([maria]));

    await expect(renderPage("nobody")).rejects.toThrow("NEXT_NOT_FOUND");
  });

  test("leaves the fetch to the client when the backend is unreachable", async () => {
    mockListExpertTemplates.mockRejectedValue(new Error("fetch failed"));

    const html = renderServerHTML(await renderPage("template-maria"));

    expect(html).not.toContain("<h1");
    expect(html).toContain("animate-pulse");
  });
});

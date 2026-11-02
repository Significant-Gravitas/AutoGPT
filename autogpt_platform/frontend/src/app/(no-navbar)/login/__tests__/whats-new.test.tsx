import type { ReactNode } from "react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { act, render, screen, within } from "@/tests/integrations/test-utils";
import { CHANGELOG_TIMEOUT_MS } from "../components/LoginWhatsNew/helpers";
import LoginPage from "../page";

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

vi.mock("../actions", () => ({
  login: vi.fn(),
}));

const CHANGELOG_URL = "https://agpt.co/docs/platform/changelog/changelog";

function release(
  dateRange: string,
  slug: string,
  highlights: string,
  title: string | null = null,
) {
  return {
    slug,
    dateRange,
    highlights,
    url: `${CHANGELOG_URL}/${slug}`,
    title,
  };
}

const RELEASES = [
  release(
    "October 8 – October 14",
    "october-8-october-14-2026",
    "Redesigned wallet, trial cancel keeps access, verified email at signup",
    "A clearer wallet, a kinder trial, and shorter replies from Otto",
  ),
  release(
    "September 24 – September 30",
    "september-24-september-30-2026",
    "A bigger roster, Clay & Rock avatars",
  ),
  release(
    "September 19",
    "september-19-2026",
    "Meet your Experts, routines",
    "Meet your Experts: hire a specialist and put it to work",
  ),
];

function serveReleases(releases: ReturnType<typeof release>[]) {
  server.use(
    http.get("/api/changelog/latest", () => HttpResponse.json({ releases })),
  );
}

async function findWhatsNew() {
  return within(await screen.findByRole("region", { name: "What's new" }));
}

describe("LoginPage What's new", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: null,
      isUserLoading: false,
      isLoggedIn: false,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("lists the latest releases by title, each linking to its changelog entry", async () => {
    serveReleases(RELEASES);
    render(<LoginPage />);
    const panel = await findWhatsNew();

    const latest = await panel.findByRole("link", {
      name: "A clearer wallet, a kinder trial, and shorter replies from Otto, October 8 – October 14, latest release (opens in a new tab)",
    });
    expect(latest.getAttribute("href")).toBe(
      `${CHANGELOG_URL}/october-8-october-14-2026`,
    );
    expect(latest.getAttribute("target")).toBe("_blank");
    expect(latest.getAttribute("rel")).toBe("noopener noreferrer");
    expect(latest.getAttribute("aria-describedby")).not.toBeNull();
    expect(
      panel.getByText(
        "Redesigned wallet, trial cancel keeps access, verified email at signup",
      ),
    ).toBeDefined();

    expect(
      panel
        .getByRole("link", {
          name: "A bigger roster, September 24 – September 30 (opens in a new tab)",
        })
        .getAttribute("href"),
    ).toBe(`${CHANGELOG_URL}/september-24-september-30-2026`);
    expect(panel.getByText("Clay & Rock avatars")).toBeDefined();
    expect(
      panel
        .getByRole("link", {
          name: "Meet your Experts: hire a specialist and put it to work, September 19 (opens in a new tab)",
        })
        .getAttribute("href"),
    ).toBe(`${CHANGELOG_URL}/september-19-2026`);
    expect(panel.getByText("Meet your Experts, routines")).toBeDefined();

    expect(panel.getAllByRole("listitem")).toHaveLength(3);
    expect(
      panel
        .getByRole("link", { name: "Full changelog (opens in a new tab)" })
        .getAttribute("href"),
    ).toBe(CHANGELOG_URL);
  });

  test("shows the releases there are when fewer than three are published", async () => {
    serveReleases([release("September 19", "september-19-2026", "Solo")]);
    render(<LoginPage />);
    const panel = await findWhatsNew();

    expect(
      await panel.findByRole("link", {
        name: "Solo, September 19, latest release (opens in a new tab)",
      }),
    ).toBeDefined();
    expect(panel.getAllByRole("listitem")).toHaveLength(1);
  });

  test.each([
    ["the changelog request fails", { error: "failed" }, 502],
    ["the changelog has no releases", { releases: [] }, 200],
  ])("links to the changelog when %s", async (_case, body, status) => {
    server.use(
      http.get("/api/changelog/latest", () =>
        HttpResponse.json(body, { status }),
      ),
    );
    render(<LoginPage />);
    const panel = await findWhatsNew();

    const fallback = await panel.findByRole("link", {
      name: /^See what we've shipped lately/,
    });
    expect(fallback.getAttribute("href")).toBe(CHANGELOG_URL);
    expect(fallback.getAttribute("target")).toBe("_blank");
    expect(panel.queryByRole("list")).toBeNull();
    expect(panel.queryByRole("link", { name: /^Full changelog/ })).toBeNull();
    expect(
      screen.getByText("Log in to your account to continue"),
    ).toBeDefined();
  });

  test("stops waiting on a changelog that never answers", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    server.use(
      http.get("/api/changelog/latest", async ({ request }) => {
        await new Promise((resolve) =>
          request.signal.addEventListener("abort", resolve),
        );
        return HttpResponse.json({ releases: [] });
      }),
    );
    render(<LoginPage />);
    const panel = await findWhatsNew();
    expect(panel.getByTestId("whats-new-skeleton")).toBeDefined();

    await act(() => vi.advanceTimersByTimeAsync(CHANGELOG_TIMEOUT_MS));

    expect(
      await panel.findByRole("link", {
        name: /^See what we've shipped lately/,
      }),
    ).toBeDefined();
    expect(panel.queryByTestId("whats-new-skeleton")).toBeNull();
  });

  test("shows markdown, long and unusual highlights as readable text", async () => {
    const longDetail =
      "a remarkably long highlight that keeps going well past what fits on a single line of the panel";
    serveReleases([
      release(
        "June 18 – July 9",
        "june-18-july-9-2026",
        `🚀 **Bold launch** (beta, invite-only), [linked docs](https://example.com), \`code_name\` &amp; more, ${longDetail}`,
      ),
    ]);
    render(<LoginPage />);
    const panel = await findWhatsNew();

    expect(
      await panel.findByRole("link", {
        name: "🚀 Bold launch (beta, invite-only), June 18 – July 9, latest release (opens in a new tab)",
      }),
    ).toBeDefined();
    const details = panel.getByText(
      `Linked docs, code_name & more, ${longDetail}`,
    );
    expect(details.getAttribute("title")).toBe(details.textContent);
  });
});

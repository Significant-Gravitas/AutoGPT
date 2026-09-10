import {
  getHireExpertMockHandler,
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
  getUpdateExpertSoulMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { getGetV1ListSystemProvidersMockHandler } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { ExpertPage as MarketplaceExpertPage } from "../[expertId]/components/ExpertPage";

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockRouterPush = vi.hoisted(() => vi.fn());
const mockParams = vi.hoisted(() => ({ expertId: "template-maria" }));
const flagStatusMock = vi.hoisted(() =>
  vi.fn(() => ({ enabled: true, ready: true })),
);

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: mockRouterPush,
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace/experts/template-maria",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => mockParams,
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? flagStatusMock()
        : actual.useFlagStatus(flag as never),
  };
});

const mariaTemplate: Expert = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: "Maria is a senior marketing strategist with fifteen years across B2B SaaS.",
  skills: ["Content strategy", "Positioning"],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [
    "The expert discloses that it is AI when acting externally.",
    "The expert asks for approval before acting externally.",
  ],
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
      integration_providers: ["anthropic", "jina"],
    },
    {
      id: "wf-2",
      name: "Automated Blog Writer",
      description: "Turns a brief into a drafted post.",
      store_listing_version_id: null,
      library_agent_id: null,
      graph_id: null,
      integration_providers: ["dataforseo", "openai"],
    },
  ],
};

const hiredMaria: Expert = {
  ...mariaTemplate,
  id: "expert-maria",
  is_template: false,
  source_template_id: "template-maria",
};

const mariaWithSamples: Expert = {
  ...mariaTemplate,
  voice_samples: [
    { label: "Punchy and bold", text: "Stop guessing what your buyers want." },
    {
      label: "Warm and story-led",
      text: "Every campaign starts with a person, not a product.",
    },
  ],
};

function renderPage() {
  return render(
    <>
      <MarketplaceExpertPage />
      <Toaster />
    </>,
  );
}

describe("Marketplace expert page", () => {
  beforeEach(() => {
    mockRouterPush.mockReset();
    mockParams.expertId = "template-maria";
    flagStatusMock.mockReturnValue({ enabled: true, ready: true });
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getGetV1ListSystemProvidersMockHandler([
        "anthropic",
        "openai",
        "jina",
        "webshare_proxy",
      ]),
    );
  });

  test("shows the profile and hires from the page", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
    );

    renderPage();

    expect(
      await screen.findByRole("heading", { level: 1, name: "Maria" }),
    ).toBeDefined();
    expect(screen.getByText("Grows your brand while you sleep")).toBeDefined();
    expect(screen.getByText("Content strategy")).toBeDefined();
    expect(
      within(screen.getByRole("region", { name: /^Workflows/ })).getByText(
        "LinkedIn Post Generator",
      ),
    ).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Back to marketplace" })
        .getAttribute("href"),
    ).toBe("/marketplace#experts");

    await userEvent.click(screen.getByRole("button", { name: "Hire Maria" }));

    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    expect(mockRouterPush).toHaveBeenCalledWith(
      `/copilot?expertId=${hiredMaria.id}&kickoff=1`,
    );
  });

  test("renders the day-one, access, plan and disclosure sections", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
    );

    renderPage();

    // Day one names the first workflow, which the backend orders
    // deterministically so the promise does not change between loads.
    const dayOne = await screen.findByRole("region", {
      name: "What Maria sets up on day one",
    });
    expect(within(dayOne).getByText("LinkedIn Post Generator")).toBeDefined();
    // The Workflows grid below carries the description; repeating it in the
    // spotlight read as duplication rather than emphasis.
    expect(
      within(dayOne).queryByText(
        "Create research-driven LinkedIn posts in minutes.",
      ),
    ).toBeNull();

    // Only what the viewer connects: Anthropic, OpenAI and Jina are on the
    // platform's own credentials and must not be asked for.
    const access = await screen.findByRole("region", {
      name: "Access Maria will ask for",
    });
    expect(within(access).getByText("DataForSEO")).toBeDefined();
    expect(within(access).queryByText("Anthropic")).toBeNull();
    expect(within(access).queryByText("OpenAI")).toBeNull();

    expect(
      screen.getByRole("heading", { name: "Included with your plan" }),
    ).toBeDefined();
    expect(screen.getByText(/capped at \$5 a week/)).toBeDefined();

    expect(
      screen.getByText(
        "The expert discloses that it is AI when acting externally.",
      ),
    ).toBeDefined();
  });

  test("drops the access and day-one sections when the data is absent", async () => {
    server.use(
      getListExpertTemplatesMockHandler([
        { ...mariaTemplate, workflows: [], protected_soul_rules: [] },
      ]),
      getListExpertsMockHandler([]),
    );

    renderPage();

    expect(
      await screen.findByRole("heading", { level: 1, name: "Maria" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("region", { name: /sets up on day one/ }),
    ).toBeNull();
    expect(
      screen.queryByRole("region", { name: /Access Maria will ask for/ }),
    ).toBeNull();
    expect(screen.queryByText(/cannot break/)).toBeNull();
    // The plan line has no data to be missing, so it always stands.
    expect(
      screen.getByRole("heading", { name: "Included with your plan" }),
    ).toBeDefined();
  });

  test("waits for the roster before offering to hire an expert already hired", async () => {
    let releaseRoster = () => {};
    const rosterHeld = new Promise<void>((resolve) => {
      releaseRoster = resolve;
    });
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      http.get("*/api/experts", async () => {
        await rosterHeld;
        return HttpResponse.json([hiredMaria]);
      }),
    );

    renderPage();

    expect(
      await screen.findByRole("heading", { level: 1, name: "Maria" }),
    ).toBeDefined();
    // The roster has not answered, so neither action may be shown yet.
    expect(screen.queryByRole("button", { name: "Hire Maria" })).toBeNull();
    expect(screen.queryByText("On your team")).toBeNull();

    releaseRoster();

    expect(await screen.findByText("On your team")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Hire Maria" })).toBeNull();
  });

  test("shows the on-your-team state with a way into the chat", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([hiredMaria]),
    );

    renderPage();

    expect(await screen.findByText("On your team")).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Chat with Maria" })
        .getAttribute("href"),
    ).toBe("/copilot?expertId=expert-maria");
    expect(screen.queryByRole("button", { name: "Hire Maria" })).toBeNull();
  });

  test("captures a voice pick as a plain-text soul PATCH after hire", async () => {
    let savedVoice = "";
    server.use(
      getListExpertTemplatesMockHandler([mariaWithSamples]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
      getUpdateExpertSoulMockHandler(async (info) => {
        const body = (await info.request.json()) as {
          voice_preferences: string;
        };
        savedVoice = body.voice_preferences;
        return { ...hiredMaria, voice_preferences: body.voice_preferences };
      }),
    );

    renderPage();

    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );

    expect(await screen.findByText("How should Maria write?")).toBeDefined();
    await userEvent.click(await screen.findByText("Punchy and bold"));
    await userEvent.click(
      screen.getByRole("button", { name: "Use this voice" }),
    );

    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    expect(savedVoice).toContain("Preferred writing style: Punchy and bold.");
    expect(savedVoice.startsWith("{")).toBe(false);
  });

  test("skips the voice pick without patching the soul", async () => {
    let soulPatched = false;
    server.use(
      getListExpertTemplatesMockHandler([mariaWithSamples]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
      getUpdateExpertSoulMockHandler(() => {
        soulPatched = true;
        return hiredMaria;
      }),
    );

    renderPage();

    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );
    expect(await screen.findByText("How should Maria write?")).toBeDefined();
    await userEvent.click(
      await screen.findByRole("button", { name: "Skip for now" }),
    );

    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    expect(soulPatched).toBe(false);
  });

  test("retries a failed voice PATCH, then celebrates and closes", async () => {
    let patchAttempts = 0;
    server.use(
      getListExpertTemplatesMockHandler([mariaWithSamples]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
      http.patch("/api/proxy/api/experts/:expertId/soul", () => {
        patchAttempts += 1;
        return patchAttempts === 1
          ? HttpResponse.json({ detail: [] }, { status: 422 })
          : HttpResponse.json(hiredMaria);
      }),
    );

    renderPage();

    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );
    expect(await screen.findByText("How should Maria write?")).toBeDefined();
    await userEvent.click(await screen.findByText("Punchy and bold"));
    await userEvent.click(
      screen.getByRole("button", { name: "Use this voice" }),
    );

    expect(await screen.findByText("Couldn't save the voice")).toBeDefined();
    expect(screen.getByText("How should Maria write?")).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: "Use this voice" }),
    );

    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText("How should Maria write?")).toBeNull(),
    );
    expect(patchAttempts).toBe(2);
  });

  test("celebrates exactly once when the voice pick is dismissed", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaWithSamples]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
    );

    renderPage();

    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );
    expect(await screen.findByText("How should Maria write?")).toBeDefined();

    await userEvent.keyboard("{Escape}");

    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText("How should Maria write?")).toBeNull(),
    );
    expect(screen.getAllByText("Maria joined your team")).toHaveLength(1);
  });

  test("shows the profile to signed-out visitors with a way to sign up", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });
    let rosterRequested = false;
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler(() => {
        rosterRequested = true;
        return [];
      }),
    );

    renderPage();

    expect(
      await screen.findByRole("heading", { level: 1, name: "Maria" }),
    ).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Get started" }).getAttribute("href"),
    ).toBe("/signup?next=%2Fmarketplace%2Fexperts%2Ftemplate-maria");
    expect(screen.queryByRole("button", { name: "Hire Maria" })).toBeNull();
    expect(screen.queryByText("Coming soon")).toBeNull();
    expect(rosterRequested).toBe(false);
  });

  test("shows the coming-soon label instead of hire actions when the flag is off", async () => {
    flagStatusMock.mockReturnValue({ enabled: false, ready: true });
    let rosterRequested = false;
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler(() => {
        rosterRequested = true;
        return [];
      }),
    );

    renderPage();

    expect(
      await screen.findByRole("heading", { level: 1, name: "Maria" }),
    ).toBeDefined();
    expect(screen.getByText("Coming soon")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Hire Maria" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Get started" })).toBeNull();
    expect(rosterRequested).toBe(false);
  });

  test("waits for the flag before showing a header action", async () => {
    flagStatusMock.mockReturnValue({ enabled: false, ready: false });
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
    );

    renderPage();

    expect(
      await screen.findByRole("heading", { level: 1, name: "Maria" }),
    ).toBeDefined();
    expect(screen.queryByText("Coming soon")).toBeNull();
    expect(screen.queryByRole("button", { name: "Hire Maria" })).toBeNull();
  });
});

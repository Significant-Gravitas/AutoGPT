import {
  getHireExpertMockHandler,
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
  getUpdateExpertSoulMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import MarketplaceExpertPage from "../[expertId]/page";

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
  protected_soul_rules: [],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  workflows: [
    {
      id: "wf-1",
      name: "LinkedIn Post Generator",
      description: "Create research-driven LinkedIn posts in minutes.",
      store_listing_version_id: null,
      library_agent_id: null,
      graph_id: null,
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
    expect(screen.getByText("LinkedIn Post Generator")).toBeDefined();
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

  test("shows a coming-soon page to signed-out visitors without fetching", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });
    let templatesRequested = false;
    server.use(
      getListExpertTemplatesMockHandler(() => {
        templatesRequested = true;
        return [mariaTemplate];
      }),
    );

    renderPage();

    expect(await screen.findByText("Coming soon")).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Sign in" }).getAttribute("href"),
    ).toBe("/login");
    expect(screen.queryByRole("heading", { name: "Maria" })).toBeNull();
    expect(templatesRequested).toBe(false);
  });

  test("shows the coming-soon page when experts are not enabled yet", async () => {
    flagStatusMock.mockReturnValue({ enabled: false, ready: true });

    renderPage();

    expect(await screen.findByText("Coming soon")).toBeDefined();
    expect(screen.queryByRole("link", { name: "Sign in" })).toBeNull();
    expect(
      screen
        .getByRole("link", { name: "Back to marketplace" })
        .getAttribute("href"),
    ).toBe("/marketplace");
  });
});

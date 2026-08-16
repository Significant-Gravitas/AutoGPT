import {
  getHireExpertMockHandler,
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
  getListExpertTemplatesMockHandler401,
  getUpdateExpertSoulMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import { screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import {
  hiredMaria,
  mariaRichTemplate,
  mariaTemplate,
  mariaWithSamples,
  renderMarketplace,
} from "./experts-section.fixtures";

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockRouterPush = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: mockRouterPush,
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
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
    useGetFlag: (flag: string) =>
      flag === "hire-experts" ? true : actual.useGetFlag(flag as never),
  };
});

describe("Marketplace ExpertsSection", () => {
  beforeEach(() => {
    mockRouterPush.mockReset();
    mockUseAuth.mockReturnValue({
      user: { id: "user-1" },
      isLoggedIn: true,
    });
  });

  test("renders experts section and hires from the profile sheet", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    await userEvent.click(await screen.findByText("Maria"));
    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );

    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    expect(await screen.findByText("View team")).toBeDefined();
    expect(screen.queryByText("Chat with Maria")).toBeNull();
    expect(mockRouterPush).toHaveBeenCalledWith(
      `/copilot?expertId=${hiredMaria.id}&kickoff=1`,
    );
  });

  test("stays hidden and fetches nothing for signed-out visitors", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });
    let templatesRequested = false;
    server.use(
      getListExpertTemplatesMockHandler(() => {
        templatesRequested = true;
        return [mariaTemplate];
      }),
      getListExpertsMockHandler([]),
    );

    renderMarketplace();

    expect(await screen.findByText("All AI Workflows")).toBeDefined();
    expect(screen.queryByText("Meet the AI Experts")).toBeNull();
    expect(templatesRequested).toBe(false);
  });

  test("shows a consistent hired state on the card and in the sheet", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([hiredMaria]),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    // Card badge reads the hired lookup.
    expect(await screen.findByText("On your team")).toBeDefined();

    await userEvent.click(await screen.findByText("Maria"));

    // Sheet reads the same lookup: hired status + an Open chat action that
    // targets the hired expert instance, not the template.
    expect(await screen.findAllByText("On your team")).toHaveLength(2);
    const chatLink = await screen.findByRole("link", { name: "Open chat" });
    expect(chatLink.getAttribute("href")).toBe(
      "/copilot?expertId=expert-maria",
    );
    expect(screen.queryByRole("button", { name: "Hire Maria" })).toBeNull();
  });

  test("hides the experts section when the templates query fails", async () => {
    server.use(
      getListExpertTemplatesMockHandler401(),
      getListExpertsMockHandler([]),
    );

    renderMarketplace();

    // Marketplace still renders; only the experts section drops out once the
    // templates query settles into its error state.
    expect(await screen.findByText("All AI Workflows")).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText("Meet the AI Experts")).toBeNull(),
    );
  });

  test("renders the profile sections from a fully populated template", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaRichTemplate]),
      getListExpertsMockHandler([]),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    await userEvent.click(await screen.findByText("Maria"));

    expect(
      await screen.findByText("What Maria sets up on day one"),
    ).toBeDefined();
    // First workflow (name + description) shows in both the day-one highlight
    // and the full list.
    expect(screen.getAllByText("Content Calendar")).toHaveLength(2);
    expect(screen.getAllByText("Plans a month of posts")).toHaveLength(2);
    expect(screen.getByText("Workflows Maria brings")).toBeDefined();
    expect(screen.getByText("SEO Audit")).toBeDefined();
    // Skill chips show on both the card and the sheet's Skills section.
    expect(screen.getAllByText("Brand strategy")).toHaveLength(2);
    expect(screen.getByText("Included with your plan")).toBeDefined();
    expect(
      screen.getByText(
        "Maria is an AI teammate. They'll always tell you before acting outside the platform.",
      ),
    ).toBeDefined();
  });

  test("renders a minimal sheet and skips empty sections", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    await userEvent.click(await screen.findByText("Maria"));

    // Always-on trust copy is present.
    expect(await screen.findByText("Included with your plan")).toBeDefined();
    expect(
      screen.getByText(
        "Maria is an AI teammate. They'll always tell you before acting outside the platform.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Hire Maria" })).toBeDefined();

    // Optional sections are omitted rather than rendered empty.
    expect(screen.queryByText("What Maria sets up on day one")).toBeNull();
    expect(screen.queryByText("Skills")).toBeNull();
    expect(screen.queryByText("Workflows Maria brings")).toBeNull();
  });

  test("expands and collapses a long expert bio", async () => {
    const longBio = "Long-form expertise. ".repeat(20);
    server.use(
      getListExpertTemplatesMockHandler([{ ...mariaTemplate, bio: longBio }]),
      getListExpertsMockHandler([]),
    );

    renderMarketplace();

    await userEvent.click(await screen.findByText("Maria"));
    const bio = await screen.findByText(longBio.trim());
    expect(bio.className).toContain("line-clamp-4");

    await userEvent.click(screen.getByRole("button", { name: "Read more" }));
    expect(screen.getByRole("button", { name: "Show less" })).toBeDefined();
    expect(bio.className).not.toContain("line-clamp-4");

    await userEvent.click(screen.getByRole("button", { name: "Show less" }));
    expect(screen.getByRole("button", { name: "Read more" })).toBeDefined();
    expect(bio.className).toContain("line-clamp-4");
  });

  test("renders a bio-only day-one section", async () => {
    const bio = "Builds a practical plan from the company context.";
    server.use(
      getListExpertTemplatesMockHandler([
        {
          ...mariaTemplate,
          bio,
          workflows: [
            {
              id: "wf-unnamed",
              store_listing_version_id: null,
              library_agent_id: null,
              graph_id: null,
              name: null,
              description: null,
            },
          ],
        },
      ]),
      getListExpertsMockHandler([]),
    );

    renderMarketplace();

    await userEvent.click(await screen.findByText("Maria"));
    expect(
      await screen.findByText("What Maria sets up on day one"),
    ).toBeDefined();
    expect(screen.getByText(bio)).toBeDefined();
    expect(screen.getAllByText("Unnamed workflow")).toHaveLength(1);
  });

  test("uses gender-neutral disclosure copy for every expert", async () => {
    const maxTemplate = {
      ...mariaTemplate,
      id: "template-max",
      name: "Max",
      role: "Sales Strategist",
    };
    server.use(
      getListExpertTemplatesMockHandler([maxTemplate]),
      getListExpertsMockHandler([]),
    );

    renderMarketplace();

    await userEvent.click(await screen.findByText("Max"));
    expect(
      await screen.findByText(
        "Max is an AI teammate. They'll always tell you before acting outside the platform.",
      ),
    ).toBeDefined();
    expect(screen.queryByText(/She'll always tell you/)).toBeNull();
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

    renderMarketplace();

    await userEvent.click(await screen.findByText("Maria"));
    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );

    // The voice pick replaces the profile before the join is celebrated.
    expect(await screen.findByText("How should Maria write?")).toBeDefined();
    await userEvent.click(await screen.findByText("Punchy and bold"));
    await userEvent.click(
      screen.getByRole("button", { name: "Use this voice" }),
    );

    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    expect(savedVoice).toContain("Preferred writing style: Punchy and bold.");
    expect(savedVoice).toContain("Stop guessing what your buyers want.");
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

    renderMarketplace();

    await userEvent.click(await screen.findByText("Maria"));
    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );
    // Wait for the picker to mount (hire awaits a list refetch first) before
    // reaching for its skip control.
    expect(await screen.findByText("How should Maria write?")).toBeDefined();
    await userEvent.click(
      await screen.findByRole("button", { name: "Skip for now" }),
    );

    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    expect(soulPatched).toBe(false);
  });

  test("refetches expert queries after the voice save so later Soul edits see it", async () => {
    let listRequests = 0;
    let voicePatched = false;
    server.use(
      getListExpertTemplatesMockHandler([mariaWithSamples]),
      getListExpertsMockHandler(() => {
        listRequests += 1;
        return voicePatched ? [hiredMaria] : [];
      }),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
      getUpdateExpertSoulMockHandler(() => {
        voicePatched = true;
        return hiredMaria;
      }),
    );

    renderMarketplace();

    await userEvent.click(await screen.findByText("Maria"));
    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );
    expect(await screen.findByText("How should Maria write?")).toBeDefined();
    const requestsBeforePick = listRequests;
    await userEvent.click(await screen.findByText("Punchy and bold"));
    await userEvent.click(
      screen.getByRole("button", { name: "Use this voice" }),
    );

    // The hire-time refetch cached the pre-voice expert as fresh for 60s; a
    // post-PATCH refetch is what keeps a follow-up Soul edit from writing the
    // stale description back over the chosen voice.
    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    await waitFor(() =>
      expect(listRequests).toBeGreaterThan(requestsBeforePick),
    );
    expect(voicePatched).toBe(true);
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

    renderMarketplace();

    await userEvent.click(await screen.findByText("Maria"));
    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );
    expect(await screen.findByText("How should Maria write?")).toBeDefined();
    await userEvent.click(await screen.findByText("Punchy and bold"));
    await userEvent.click(
      screen.getByRole("button", { name: "Use this voice" }),
    );

    // The hire already succeeded, so the picker stays open to retry rather
    // than losing the choice.
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

  test("celebrates exactly once when the completed hire is dismissed", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaWithSamples]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
    );

    renderMarketplace();

    await userEvent.click(await screen.findByText("Maria"));
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
});

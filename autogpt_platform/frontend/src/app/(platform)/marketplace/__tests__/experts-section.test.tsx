import {
  getHireExpertMockHandler,
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
  getUpdateExpertSoulMockHandler,
  getUpdateExpertSoulMockHandler422,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

const mockUseAuth = vi.hoisted(() => vi.fn());

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

const mariaTemplate: Expert = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [
    "The expert discloses that it is AI when acting externally.",
    "External actions require approval.",
  ],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  workflows: [],
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

function renderMarketplace() {
  return render(
    <>
      <MainMarkeplacePage />
      <Toaster />
    </>,
  );
}

describe("Marketplace ExpertsSection", () => {
  beforeEach(() => {
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
    expect(await screen.findByText("Chat with Maria")).toBeDefined();
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

  test("keeps the voice pick open when the soul PATCH fails", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaWithSamples]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
      getUpdateExpertSoulMockHandler422(),
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
  });

  test("hired template shows hired state", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([hiredMaria]),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    expect(await screen.findByText("Hired")).toBeDefined();
  });
});

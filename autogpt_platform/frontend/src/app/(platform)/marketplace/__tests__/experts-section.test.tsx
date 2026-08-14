import {
  getHireExpertMockHandler,
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
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

  test("hired template shows hired state", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([hiredMaria]),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    expect(await screen.findByText("Hired")).toBeDefined();
  });

  test("emits funnel events when opening a profile and starting a hire", async () => {
    const funnelBodies: { type: string; data: Record<string, unknown> }[] = [];
    server.use(
      http.post(/log_raw_analytics/, async ({ request }) => {
        funnelBodies.push(
          (await request.json()) as {
            type: string;
            data: Record<string, unknown>;
          },
        );
        return HttpResponse.json({ status: "ok" });
      }),
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
    );

    renderMarketplace();

    await userEvent.click(await screen.findByText("Maria"));
    await waitFor(() =>
      expect(funnelBodies.map((b) => b.type)).toContain(
        "expert_profile_opened",
      ),
    );
    expect(
      funnelBodies.find((b) => b.type === "expert_profile_opened")?.data,
    ).toEqual({ template_id: mariaTemplate.id });

    await userEvent.click(
      await screen.findByRole("button", { name: "Hire Maria" }),
    );
    await waitFor(() =>
      expect(funnelBodies.map((b) => b.type)).toContain("hire_started"),
    );
  });
});

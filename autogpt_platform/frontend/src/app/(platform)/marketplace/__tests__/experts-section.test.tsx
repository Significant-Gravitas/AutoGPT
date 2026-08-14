import {
  getHireExpertMockHandler,
  getListExpertsMockHandler,
  getListExpertsMockHandler401,
  getListExpertTemplatesMockHandler,
  getListExpertTemplatesMockHandler401,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
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

const mariaRichTemplate: Expert = {
  ...mariaTemplate,
  bio: "Maria has run brand launches for a decade and loves a tidy funnel.",
  skills: ["Brand strategy", "SEO"],
  workflows: [
    {
      id: "wf-1",
      store_listing_version_id: "slv-1",
      library_agent_id: "lib-1",
      graph_id: "graph-1",
      name: "Content Calendar",
      description: "Plans a month of posts",
    },
    {
      id: "wf-2",
      store_listing_version_id: "slv-2",
      library_agent_id: "lib-2",
      graph_id: "graph-2",
      name: "SEO Audit",
      description: null,
    },
  ],
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

  test("keeps the hire action gated while the team lookup is unresolved", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler(() => new Promise<Expert[]>(() => {})),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    // The card shows a placeholder instead of claiming "Hire" or "On your
    // team" while the hired lookup is unresolved.
    expect(screen.queryByText("Hire")).toBeNull();
    expect(screen.queryByText("On your team")).toBeNull();

    await userEvent.click(await screen.findByText("Maria"));

    const hireButton = await screen.findByRole("button", {
      name: "Hire Maria",
    });
    expect(hireButton.hasAttribute("disabled")).toBe(true);
  });

  test("offers a retryable team status instead of hiring when the lookup fails", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler401(),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    expect(screen.queryByText("Hire")).toBeNull();
    expect(screen.queryByText("On your team")).toBeNull();

    await userEvent.click(await screen.findByText("Maria"));

    expect(
      await screen.findByText("Team status unavailable right now."),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Hire Maria" })).toBeNull();

    server.use(getListExpertsMockHandler([hiredMaria]));
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));

    // After a successful retry the sheet resolves to the real hired state.
    expect(
      await screen.findByRole("link", { name: "Open chat" }),
    ).toBeDefined();
  });

  test("day-one highlight skips workflows without displayable names", async () => {
    const templateWithDanglingRef: Expert = {
      ...mariaTemplate,
      workflows: [
        {
          id: "wf-dangling",
          store_listing_version_id: null,
          library_agent_id: null,
          graph_id: null,
          name: null,
          description: null,
        },
        {
          id: "wf-named",
          store_listing_version_id: "slv-2",
          library_agent_id: null,
          graph_id: null,
          name: "SEO Audit",
          description: null,
        },
      ],
    };
    server.use(
      getListExpertTemplatesMockHandler([templateWithDanglingRef]),
      getListExpertsMockHandler([]),
    );

    renderMarketplace();

    await userEvent.click(await screen.findByText("Maria"));

    expect(
      await screen.findByText("What Maria sets up on day one"),
    ).toBeDefined();
    // The named workflow is promised (highlight + list); the dangling ref
    // only appears in the full list, never as the day-one promise.
    expect(screen.getAllByText("SEO Audit")).toHaveLength(2);
    expect(screen.getAllByText("Unnamed workflow")).toHaveLength(1);
  });
});

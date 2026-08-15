import {
  getHireExpertMockHandler,
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
  getListExpertTemplatesMockHandler401,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import { screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import {
  hiredMaria,
  mariaRichTemplate,
  mariaTemplate,
  renderMarketplace,
} from "./experts-section.fixtures";

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
});

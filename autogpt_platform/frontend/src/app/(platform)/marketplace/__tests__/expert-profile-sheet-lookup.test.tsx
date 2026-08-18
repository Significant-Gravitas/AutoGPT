import {
  getListExpertsMockHandler,
  getListExpertsMockHandler401,
  getListExpertTemplatesMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { ExpertProfileActions } from "../components/ExpertsSection/components/ExpertProfileSheet/components/ExpertProfileActions/ExpertProfileActions";
import {
  hiredMaria,
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

describe("Expert profile sheet lookup states", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: { id: "user-1" },
      isLoggedIn: true,
    });
  });

  test("keeps the hire action gated while the team lookup is unresolved", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler(() => new Promise<Expert[]>(() => {})),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    expect(screen.queryByText("Hire")).toBeNull();
    expect(screen.queryByText("On your team")).toBeNull();
    expect(await screen.findByLabelText("Loading team status")).toBeDefined();

    await userEvent.click(await screen.findByText("Maria"));

    const hireButton = await screen.findByRole("button", {
      name: "Checking your team…",
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
    expect(await screen.findByText("Team status unavailable")).toBeDefined();
    expect(screen.getAllByText("Team status unavailable")).toHaveLength(1);
    expect(screen.getByText("View details")).toBeDefined();

    await userEvent.click(await screen.findByText("Maria"));

    expect(
      await screen.findByText("Team status unavailable right now."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Hire Maria" })).toBeDefined();

    server.use(getListExpertsMockHandler([hiredMaria]));
    await userEvent.click(screen.getByRole("button", { name: "Retry" }));

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
    expect(screen.getAllByText("SEO Audit")).toHaveLength(2);
    expect(screen.getAllByText("Unnamed workflow")).toHaveLength(1);
  });

  test("links to the team when a hired expert ID is unavailable", () => {
    render(
      <ExpertProfileActions
        expertName="Maria"
        isHired
        isHiring={false}
        onHire={vi.fn()}
        hiredExpertId={null}
        hiredLookup="loaded"
        onRetryHiredLookup={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("link", { name: "View team" }).getAttribute("href"),
    ).toBe("/team");
  });
});

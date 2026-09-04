import {
  getGetExpertMockHandler,
  getListExpertCredentialsMockHandler,
  getListExpertRunsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ExpertDetailPage from "../page";

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: () => ({ enabled: true, ready: true }),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/team/expert-maria",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ expertId: "expert-maria" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const maria = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  color: "",
  role: "Marketing Strategist",
  tagline: null,
  bio: "Maria is a senior marketing strategist.",
  skills: [],
  identity: "You are Maria.",
  voice_preferences: "Direct.",
  voice_samples: [],
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
  weekly_budget: null,
  weekly_spend: 0,
  schedules_paused_at: null,
  pod_id: null,
} as unknown as Expert;

const linkedin: ExpertCredentialRef = {
  credential_id: "cred-linkedin",
  provider: "linkedin",
  title: "Work LinkedIn",
  type: "oauth2",
};

beforeEach(() => {
  server.use(
    getGetExpertMockHandler(maria),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertRunsMockHandler([]),
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json([
        {
          id: "cred-linkedin",
          provider: "linkedin",
          type: "oauth2",
          title: "Work LinkedIn",
        },
        {
          id: "cred-notion",
          provider: "notion",
          type: "api_key",
          title: "Team Notion",
        },
      ]),
    ),
  );
});

async function openIntegrationsTab() {
  await userEvent.click(
    await screen.findByRole("tab", { name: /integrations/i }),
  );
}

describe("managing an expert's integrations", () => {
  it("lists what the expert can reach", async () => {
    server.use(getListExpertCredentialsMockHandler([linkedin]));

    render(<ExpertDetailPage />);

    await openIntegrationsTab();

    const section = await screen.findByTestId("expert-integrations-section");
    expect(await within(section).findByText("Work LinkedIn")).toBeDefined();
    expect(within(section).getByText("LinkedIn")).toBeDefined();
  });

  it("titles the tab and filters integrations by search", async () => {
    server.use(getListExpertCredentialsMockHandler([linkedin]));

    render(<ExpertDetailPage />);

    await openIntegrationsTab();

    const section = await screen.findByTestId("expert-integrations-section");
    expect(within(section).getByText("Maria's Integrations")).toBeDefined();
    await within(section).findByText("Work LinkedIn");

    await userEvent.type(
      within(section).getByRole("searchbox", { name: "Search integrations" }),
      "notion",
    );
    expect(within(section).getByText("No integrations match.")).toBeDefined();

    await userEvent.clear(
      within(section).getByRole("searchbox", { name: "Search integrations" }),
    );
    expect(within(section).getByText("Work LinkedIn")).toBeDefined();
  });

  it("explains the empty state instead of showing a bare list", async () => {
    server.use(getListExpertCredentialsMockHandler([]));

    render(<ExpertDetailPage />);

    await openIntegrationsTab();

    const section = await screen.findByTestId("expert-integrations-section");
    expect(
      await within(section).findByText(
        /Nothing connected yet\. Add a tool and Maria can use it/,
      ),
    ).toBeDefined();
  });

  it("revokes an integration through the API", async () => {
    let revoked: string | null = null;
    server.use(
      getListExpertCredentialsMockHandler([linkedin]),
      http.delete(
        "*/api/experts/expert-maria/credentials/:credentialId",
        ({ params }) => {
          revoked = params.credentialId as string;
          return HttpResponse.json([]);
        },
      ),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();

    await userEvent.click(
      await screen.findByRole("button", { name: "Remove Work LinkedIn" }),
    );

    await waitFor(() => expect(revoked).toBe("cred-linkedin"));
  });

  it("only offers credentials the expert does not already have", async () => {
    let granted: string[] = [];
    server.use(
      getListExpertCredentialsMockHandler([linkedin]),
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "notion",
            description: "Docs and databases",
            supported_auth_types: ["api_key"],
          },
          {
            name: "linkedin",
            description: "Professional network",
            supported_auth_types: ["oauth2"],
          },
        ]),
      ),
      http.post(
        "*/api/experts/expert-maria/credentials",
        async ({ request }) => {
          const body = (await request.json()) as { credential_ids: string[] };
          granted = body.credential_ids;
          return HttpResponse.json([linkedin]);
        },
      ),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();
    await userEvent.click(
      await screen.findByRole("button", { name: /Use existing/ }),
    );

    const dialog = await screen.findByRole("dialog");
    const list = within(dialog).getByRole("list", {
      name: "Existing connections",
    });
    const useNotion = within(list).getByRole("button", { name: /Notion/ });
    expect(within(list).queryByRole("button", { name: /LinkedIn/ })).toBeNull();

    await userEvent.click(useNotion);
    await waitFor(() => expect(granted).toEqual(["cred-notion"]));
  });

  it("offers connecting a new service when there is nothing left to grant", async () => {
    server.use(
      getListExpertCredentialsMockHandler([]),
      http.get("*/api/integrations/providers", () => HttpResponse.json([])),
      http.get("*/api/integrations/credentials", () => HttpResponse.json([])),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();

    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );

    expect(await screen.findByLabelText("Search services")).toBeDefined();
    expect(
      screen.getByText(
        "Pick a service to connect. Maria will be able to use it on your behalf.",
      ),
    ).toBeDefined();
  });

  it("grants an existing credential from the Use existing dialog", async () => {
    let granted: string[] = [];
    server.use(
      getListExpertCredentialsMockHandler([linkedin]),
      http.post(
        "*/api/experts/expert-maria/credentials",
        async ({ request }) => {
          const body = (await request.json()) as { credential_ids: string[] };
          granted = body.credential_ids;
          return HttpResponse.json([]);
        },
      ),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();
    await userEvent.click(
      await screen.findByRole("button", { name: /Use existing/ }),
    );

    const dialog = await screen.findByRole("dialog");
    const list = within(dialog).getByRole("list", {
      name: "Existing connections",
    });
    expect(within(list).getByText("Team Notion")).toBeDefined();
    expect(within(list).queryByText("Work LinkedIn")).toBeNull();
    await userEvent.click(within(list).getByRole("button", { name: /Notion/ }));

    await waitFor(() => expect(granted).toEqual(["cred-notion"]));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("asks which credential to use when a provider has several", async () => {
    let granted: string[] = [];
    server.use(
      getListExpertCredentialsMockHandler([linkedin]),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json([
          {
            id: "cred-notion",
            provider: "notion",
            type: "api_key",
            title: "Team Notion",
          },
          {
            id: "cred-notion-2",
            provider: "notion",
            type: "api_key",
            title: "Personal Notion",
          },
        ]),
      ),
      http.post(
        "*/api/experts/expert-maria/credentials",
        async ({ request }) => {
          const body = (await request.json()) as { credential_ids: string[] };
          granted = body.credential_ids;
          return HttpResponse.json([]);
        },
      ),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();
    await userEvent.click(
      await screen.findByRole("button", { name: /Use existing/ }),
    );
    const dialog = await screen.findByRole("dialog");
    expect(
      within(dialog).getByText("2 connections · choose one"),
    ).toBeDefined();
    await userEvent.click(
      within(dialog).getByRole("button", { name: /Notion/ }),
    );

    const choices = await within(dialog).findByRole("list", {
      name: "Notion connections",
    });
    expect(within(choices).getAllByRole("listitem")).toHaveLength(2);
    await userEvent.click(
      within(choices).getByRole("button", {
        name: "Let Maria use Personal Notion",
      }),
    );

    await waitFor(() => expect(granted).toEqual(["cred-notion-2"]));
  });

  it("opens a separate dialog for each header button", async () => {
    server.use(
      getListExpertCredentialsMockHandler([linkedin]),
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "notion",
            description: "Docs and databases",
            supported_auth_types: ["api_key"],
          },
        ]),
      ),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();
    await userEvent.click(
      await screen.findByRole("button", { name: /Use existing/ }),
    );
    let dialog = await screen.findByRole("dialog");
    expect(
      within(dialog).getByRole("list", { name: "Existing connections" }),
    ).toBeDefined();
    expect(within(dialog).queryByLabelText("Search services")).toBeNull();
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    await userEvent.click(
      screen.getByRole("button", { name: /Add integration/ }),
    );
    dialog = await screen.findByRole("dialog");
    expect(
      await within(dialog).findByLabelText("Search services"),
    ).toBeDefined();
    expect(await within(dialog).findByText("Notion")).toBeDefined();
    expect(within(dialog).queryByText(/Let Maria use/)).toBeNull();
  });

  it("grants only the credential the dialog created", async () => {
    const workLinkedin = {
      id: "cred-linkedin",
      provider: "linkedin",
      type: "oauth2",
      title: "Work LinkedIn",
    };
    const teamNotion = {
      id: "cred-notion",
      provider: "notion",
      type: "api_key",
      title: "Team Notion",
    };
    // Lands on the account while the dialog is open, e.g. from another tab.
    const teamSlack = {
      id: "cred-slack",
      provider: "slack",
      type: "oauth2",
      title: "Team Slack",
    };
    let connected = [workLinkedin];
    const granted: string[][] = [];

    server.use(
      getListExpertCredentialsMockHandler([linkedin]),
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "notion",
            description: "Docs and databases",
            supported_auth_types: ["api_key"],
          },
        ]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(connected),
      ),
      http.post("*/api/integrations/notion/credentials", () => {
        connected = [workLinkedin, teamSlack, teamNotion];
        return HttpResponse.json(teamNotion, { status: 201 });
      }),
      http.post(
        "*/api/experts/expert-maria/credentials",
        async ({ request }) => {
          const body = (await request.json()) as { credential_ids: string[] };
          granted.push(body.credential_ids);
          return HttpResponse.json([linkedin]);
        },
      ),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();

    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );
    await userEvent.click(await screen.findByText("Notion"));

    expect(await screen.findByText("Connect AutoGPT to Notion")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: /API Key/ }));
    await userEvent.type(
      await screen.findByPlaceholderText("My Notion key"),
      "Team Notion",
    );
    await userEvent.type(screen.getByPlaceholderText("sk-..."), "secret-value");
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => expect(granted).toEqual([["cred-notion"]]));
  });

  it("does not offer integrations when the expert's own list fails to load", async () => {
    let grantAttempts = 0;
    server.use(
      http.get("*/api/experts/expert-maria/credentials", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
      http.post("*/api/experts/expert-maria/credentials", () => {
        grantAttempts += 1;
        return HttpResponse.json([]);
      }),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();

    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).queryByText(/Let Maria use/)).toBeNull();
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(
      screen
        .getByRole("button", { name: /Use existing/ })
        .hasAttribute("disabled"),
    ).toBe(true);
    expect(grantAttempts).toBe(0);
  });

  it("names an MCP credential after the service behind it", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        {
          credential_id: "cred-mcp",
          provider: "mcp",
          title: "MCP: mcp.sentry.dev",
          type: "host_scoped",
        },
      ]),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();

    const section = await screen.findByTestId("expert-integrations-section");
    expect(await within(section).findByText("Sentry")).toBeDefined();
    expect(within(section).getByText("MCP server")).toBeDefined();
    expect(within(section).getByText("Ready")).toBeDefined();
  });
});

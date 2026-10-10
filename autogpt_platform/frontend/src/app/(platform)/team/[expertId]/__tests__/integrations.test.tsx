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

vi.mock("@/components/contextual/DeviceAuth/DeviceAuthConnectButton", () => ({
  DeviceAuthConnectButton: ({
    onSuccess,
  }: {
    onSuccess: (credential: {
      id: string;
      provider: string;
      type: "oauth2";
      title: string;
    }) => void;
  }) => (
    <button
      onClick={() =>
        onSuccess({
          id: "cred-device",
          provider: "openai",
          type: "oauth2",
          title: "Device account",
        })
      }
    >
      Complete device approval
    </button>
  ),
}));

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
  service: "linkedin",
  service_name: null,
  service_icon: "linkedin",
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

  it("revokes an integration through the API and refreshes the expert", async () => {
    let revoked: string | null = null;
    let expertReads = 0;
    server.use(
      getGetExpertMockHandler(() => {
        expertReads += 1;
        return maria;
      }),
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
    // The header logos come from the expert, so it is read again.
    await waitFor(() => expect(expertReads).toBe(2));
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

  it("filters the connect dialog's services by connection state", async () => {
    server.use(
      getListExpertCredentialsMockHandler([]),
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "notion",
            description: "Docs and databases",
            supported_auth_types: ["api_key"],
          },
          {
            name: "slack",
            description: "Team chat",
            supported_auth_types: ["oauth2"],
          },
        ]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json([
          {
            id: "cred-slack",
            provider: "slack",
            type: "oauth2",
            title: "Team Slack",
          },
        ]),
      ),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();
    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );
    const dialog = await screen.findByRole("dialog");
    const list = await within(dialog).findByRole("list", { name: "Services" });
    expect(within(list).getByRole("button", { name: /Notion/ })).toBeDefined();
    expect(within(list).getByRole("button", { name: /Slack/ })).toBeDefined();

    const filters = within(dialog).getByRole("group", {
      name: "Filter services",
    });
    await userEvent.click(
      within(filters).getByRole("button", { name: "Connected" }),
    );
    expect(within(list).queryByRole("button", { name: /Notion/ })).toBeNull();
    expect(within(list).getByRole("button", { name: /Slack/ })).toBeDefined();

    await userEvent.click(
      within(filters).getByRole("button", { name: "Not connected" }),
    );
    expect(within(list).getByRole("button", { name: /Notion/ })).toBeDefined();
    expect(within(list).queryByRole("button", { name: /Slack/ })).toBeNull();
  });

  it("offers a vendor once and opens its sign-in first, with an API key alternative", async () => {
    server.use(
      getListExpertCredentialsMockHandler([]),
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "linear",
            description: "Issues and projects",
            supported_auth_types: ["api_key"],
            service: "linear",
            service_name: null,
            service_icon: "linear",
          },
          {
            name: "mcp_linear",
            display_name: "Linear",
            supported_auth_types: [],
            service: "linear",
            service_name: "Linear",
            service_icon: "linear",
            mcp_server: {
              server_url: "https://mcp.linear.app/mcp",
              documentation_url: "https://linear.app/docs",
              setup_instructions: "Sign in to Linear.",
              connection_mode: "hosted",
              auth_methods: ["oauth"],
            },
          },
          {
            name: "mcp_sentry",
            display_name: "Sentry",
            supported_auth_types: [],
            service: "sentry",
            service_name: "Sentry",
            service_icon: "sentry",
            mcp_server: {
              server_url: "https://mcp.sentry.dev/mcp",
              documentation_url: "https://docs.sentry.io",
              setup_instructions: "Sign in to Sentry.",
              connection_mode: "hosted",
              auth_methods: ["oauth"],
            },
          },
        ]),
      ),
    );

    render(<ExpertDetailPage />);
    await openIntegrationsTab();
    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );
    const dialog = await screen.findByRole("dialog");
    const list = await within(dialog).findByRole("list", { name: "Services" });

    expect(
      within(list).getAllByRole("button", { name: /Linear/ }),
    ).toHaveLength(1);
    expect(within(list).getByRole("button", { name: /Sentry/ })).toBeDefined();
    expect(within(dialog).queryByText("MCP")).toBeNull();

    await userEvent.click(within(list).getByRole("button", { name: /Linear/ }));
    expect(await within(dialog).findByText("Sign in to Linear.")).toBeDefined();
    expect(
      within(dialog).getByRole("button", { name: "More ways to connect" }),
    ).toBeDefined();

    await userEvent.click(
      within(dialog).getByRole("button", { name: "More ways to connect" }),
    );
    expect(
      await within(dialog).findByRole("button", { name: /API Key/ }),
    ).toBeDefined();
    expect(
      within(dialog).getByRole("button", {
        name: "Use the Linear sign-in instead",
      }),
    ).toBeDefined();
  });

  it("grants a credential connected through the service sign-in", async () => {
    let granted: string[] = [];
    server.use(
      getListExpertCredentialsMockHandler([]),
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "mcp_sentry",
            display_name: "Sentry",
            supported_auth_types: [],
            service: "sentry",
            service_name: "Sentry",
            service_icon: "sentry",
            mcp_server: {
              server_url: "https://mcp.sentry.dev/mcp",
              documentation_url: "https://docs.sentry.io",
              setup_instructions: "Paste your Sentry token.",
              connection_mode: "hosted",
              auth_methods: ["bearer"],
            },
          },
        ]),
      ),
      http.post("*/api/mcp/discover-tools", () =>
        HttpResponse.json({
          tools: [],
          server_url: "https://mcp.sentry.dev/mcp",
        }),
      ),
      http.post("*/api/mcp/token", () =>
        HttpResponse.json({
          id: "cred-sentry",
          provider: "mcp",
          type: "oauth2",
          title: "MCP: mcp.sentry.dev",
          host: "https://mcp.sentry.dev/mcp",
          service: "sentry",
          service_name: "Sentry",
          service_icon: "sentry",
        }),
      ),
      http.post(
        "*/api/experts/expert-maria/credentials",
        async ({ request }) => {
          const body = (await request.json()) as { credential_ids: string[] };
          granted = body.credential_ids;
          return HttpResponse.json([
            {
              credential_id: "cred-sentry",
              provider: "mcp",
              title: "MCP: mcp.sentry.dev",
              type: "oauth2",
              service: "sentry",
              service_name: "Sentry",
              service_icon: "sentry",
            },
          ]);
        },
      ),
    );

    render(<ExpertDetailPage />);
    await openIntegrationsTab();
    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );
    const dialog = await screen.findByRole("dialog");
    await userEvent.click(
      within(
        await within(dialog).findByRole("list", { name: "Services" }),
      ).getByRole("button", { name: /Sentry/ }),
    );
    await userEvent.type(
      await within(dialog).findByLabelText("API token"),
      "sntrys_token_value",
    );
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Save token" }),
    );

    await waitFor(() => expect(granted).toEqual(["cred-sentry"]));
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

  it("shows the credential the dialog created without a reload", async () => {
    const teamNotion = {
      id: "cred-notion",
      provider: "notion",
      type: "api_key",
      title: "Team Notion",
    };
    const notionRef: ExpertCredentialRef = {
      credential_id: "cred-notion",
      provider: "notion",
      title: "Team Notion",
      type: "api_key",
    };
    let granted: ExpertCredentialRef[] = [linkedin];
    const seoAudit = {
      id: "wf-1",
      store_listing_version_id: "slv-1",
      library_agent_id: "lib-1",
      graph_id: "graph-1",
      name: "SEO Audit",
      description: null,
      schedule_cron: "0 9 * * 1",
      schedule_id: null,
    };

    server.use(
      // The grant is what lets the backend create the schedule this
      // workflow was waiting on, so the expert itself changes too.
      getGetExpertMockHandler(() => ({
        ...maria,
        workflows: [
          granted.length > 1
            ? { ...seoAudit, schedule_id: "sched-1" }
            : seoAudit,
        ],
      })),
      http.get("*/api/experts/expert-maria/credentials", () =>
        HttpResponse.json(granted),
      ),
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "notion",
            description: "Docs and databases",
            supported_auth_types: ["api_key"],
          },
        ]),
      ),
      http.post("*/api/integrations/notion/credentials", () =>
        HttpResponse.json(teamNotion, { status: 201 }),
      ),
      http.post("*/api/experts/expert-maria/credentials", () => {
        granted = [linkedin, notionRef];
        return HttpResponse.json(granted);
      }),
    );

    render(<ExpertDetailPage />);

    await openIntegrationsTab();
    const section = await screen.findByTestId("expert-integrations-section");
    await within(section).findByText("Work LinkedIn");

    await userEvent.click(
      screen.getByRole("button", { name: /Add integration/ }),
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

    expect(await within(section).findByText("Team Notion")).toBeDefined();

    await userEvent.click(screen.getByRole("tab", { name: /workflows/i }));
    expect(await screen.findByText("Scheduled")).toBeDefined();
    expect(screen.queryByText("Needs setup")).toBeNull();
  });

  it("grants the approved device credential to the expert", async () => {
    const granted: string[][] = [];
    server.use(
      getListExpertCredentialsMockHandler([]),
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "openai",
            description: "Device account",
            supported_auth_types: ["device_code"],
          },
        ]),
      ),
      http.post(
        "*/api/experts/expert-maria/credentials",
        async ({ request }) => {
          const body = (await request.json()) as { credential_ids: string[] };
          granted.push(body.credential_ids);
          return HttpResponse.json([]);
        },
      ),
    );
    render(<ExpertDetailPage />);
    await openIntegrationsTab();
    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );
    await userEvent.click(await screen.findByText("OpenAI"));
    await userEvent.click(
      await screen.findByRole("button", { name: /Device/ }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Complete device approval" }),
    );
    await waitFor(() => expect(granted).toEqual([["cred-device"]]));
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

  it("files an MCP credential under its service, with no MCP wording", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        {
          credential_id: "cred-mcp",
          provider: "mcp",
          title: "MCP: mcp.sentry.dev",
          type: "oauth2",
          service: "sentry",
          service_name: "Sentry",
          service_icon: "sentry",
        },
      ]),
    );

    render(<ExpertDetailPage />);
    await openIntegrationsTab();

    const section = await screen.findByTestId("expert-integrations-section");
    expect(await within(section).findByText("Sentry")).toBeDefined();
    expect(within(section).queryByText(/MCP/)).toBeNull();
    expect(within(section).getByText("Ready")).toBeDefined();
    expect(
      within(section)
        .getByRole("img", { name: "Sentry logo" })
        .getAttribute("src"),
    ).toContain("sentry.png");
  });

  it("files a self-hosted MCP credential under its hostname", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        {
          credential_id: "cred-mcp-custom",
          provider: "mcp",
          title: "MCP: mcp.internal.example",
          type: "oauth2",
          service: "mcp:mcp.internal.example",
          service_name: "mcp.internal.example",
          service_icon: null,
        },
      ]),
    );

    render(<ExpertDetailPage />);
    await openIntegrationsTab();

    const section = await screen.findByTestId("expert-integrations-section");
    expect(
      await within(section).findAllByText("mcp.internal.example"),
    ).toHaveLength(2);
    expect(
      within(section).getByRole("img", { name: "mcp.internal.example logo" }),
    ).toBeDefined();
    expect(within(section).queryByText(/MCP server/)).toBeNull();
  });

  it("groups an MCP credential and an API key for the same vendor together", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        {
          credential_id: "cred-linear-key",
          provider: "linear",
          title: "Linear key",
          type: "api_key",
          service: "linear",
          service_name: null,
          service_icon: "linear",
        },
        {
          credential_id: "cred-linear-mcp",
          provider: "mcp",
          title: "MCP: mcp.linear.app",
          type: "oauth2",
          service: "linear",
          service_name: "Linear",
          service_icon: "linear",
        },
        {
          credential_id: "cred-posthog-key",
          provider: "posthog",
          title: "PostHog key",
          type: "api_key",
          service: "posthog",
          service_name: null,
          service_icon: null,
        },
        {
          credential_id: "cred-posthog-mcp",
          provider: "mcp",
          title: "MCP: mcp.posthog.com",
          type: "oauth2",
          service: "posthog",
          service_name: "PostHog",
          service_icon: "posthog",
        },
      ]),
    );

    render(<ExpertDetailPage />);
    await openIntegrationsTab();

    const section = await screen.findByTestId("expert-integrations-section");
    const rows = await within(section).findAllByTestId(
      "expert-integration-row",
    );
    expect(rows).toHaveLength(4);
    expect(within(section).getAllByText("Linear")).toHaveLength(1);
    expect(within(section).getAllByText("PostHog")).toHaveLength(1);
    expect(within(section).queryByText("Posthog")).toBeNull();
  });
});

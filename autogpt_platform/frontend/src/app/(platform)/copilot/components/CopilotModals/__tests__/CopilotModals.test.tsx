import {
  getGetV1ListCredentialsMockHandler,
  getGetV1ListProvidersMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import {
  getGetV1ListExecutionSchedulesForAUserMockHandler,
  getListCopilotFollowupSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { ConnectServiceDialog } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/ConnectServiceDialog";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { useCopilotModal } from "../../../useCopilotModal";
import { CopilotModals } from "../CopilotModals";

function Harness() {
  const { openModal } = useCopilotModal();

  function openSkills() {
    openModal("skills");
  }

  function openScheduled() {
    openModal("scheduled");
  }

  function openIntegrations() {
    openModal("integrations");
  }

  function openConnect() {
    openModal("connect");
  }

  return (
    <>
      <button onClick={openSkills}>open-skills</button>
      <button onClick={openScheduled}>open-scheduled</button>
      <button onClick={openIntegrations}>open-integrations</button>
      <button onClick={openConnect}>open-connect</button>
      <CopilotModals />
    </>
  );
}

const linearBlockAndSignIn = getGetV1ListProvidersMockHandler([
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
]);

async function openLinear(dialog: HTMLElement) {
  await userEvent.click(
    await within(dialog).findByRole("button", { name: /Linear/ }),
  );
}

function arrangeSentryGrantForMaria() {
  const grants = { granted: [] as string[] };
  useCopilotUIStore.setState({
    contextPanelExpert: { id: "expert-maria", name: "Maria" },
  });
  server.use(
    getGetV1ListProvidersMockHandler([
      {
        name: "mcp_sentry",
        display_name: "Sentry",
        description: "Errors",
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
          oauth_write_scopes: [],
          server_url_options: [],
          allow_custom_url: false,
        },
      },
    ]),
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
        service: "sentry",
      }),
    ),
    http.post("*/api/experts/expert-maria/credentials", async ({ request }) => {
      grants.granted = (
        (await request.json()) as { credential_ids: string[] }
      ).credential_ids;
      return HttpResponse.json([]);
    }),
  );
  return grants;
}

describe("CopilotModals", () => {
  beforeEach(() => {
    useCopilotUIStore.setState({
      initialPrompt: null,
      contextPanelExpert: null,
    });
    server.use(
      getListCopilotSkillsMockHandler([]),
      getListCopilotFollowupSchedulesMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([]),
      getGetV1ListCredentialsMockHandler([]),
    );
  });

  afterEach(() => {
    server.resetHandlers();
  });

  test("no modal renders by default", () => {
    render(<Harness />);
    expect(screen.queryByText("Skills")).toBeNull();
    expect(screen.queryByText("Scheduled")).toBeNull();
    expect(screen.queryByText("Integrations")).toBeNull();
  });

  test("opens the skills modal with header actions and empty state", async () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("open-skills"));

    expect(await screen.findByText("Skills")).toBeDefined();
    expect(await screen.findByTestId("skills-empty")).toBeDefined();
    expect(screen.getByTestId("skill-new-button")).toBeDefined();
    expect(screen.getByTestId("skill-upload-button")).toBeDefined();
  });

  test("New skill closes the modal and prefills the composer store", async () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("open-skills"));

    fireEvent.click(await screen.findByTestId("skill-new-button"));

    await vi.waitFor(() => {
      expect(useCopilotUIStore.getState().initialPrompt).toContain(
        "I want to teach you a new skill",
      );
    });
    await vi.waitFor(() => {
      expect(screen.queryByText("Skills")).toBeNull();
    });
  });

  test("opens the scheduled modal and New scheduled task prefills the store", async () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("open-scheduled"));

    expect(await screen.findByTestId("followups-empty")).toBeDefined();
    fireEvent.click(screen.getByTestId("schedule-new-button"));

    await vi.waitFor(() => {
      expect(useCopilotUIStore.getState().initialPrompt).toContain(
        "I want to create a new scheduled task",
      );
    });
  });

  test("opens the integrations modal with the Connect Service action", async () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("open-integrations"));

    expect(await screen.findByText("Integrations")).toBeDefined();
    expect(
      (await screen.findAllByText("Connect Service")).length,
    ).toBeGreaterThan(0);
  });

  test("opens the connect dialog directly, without the credentials list", async () => {
    server.use(
      getGetV1ListProvidersMockHandler([
        {
          name: "github",
          description: "Issues and PRs",
          supported_auth_types: ["oauth2", "api_key"],
        },
      ]),
    );
    render(<Harness />);
    fireEvent.click(screen.getByText("open-connect"));

    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("Issues and PRs")).toBeDefined();
    expect(screen.queryByText("No integration connected")).toBeNull();
  });

  test("the connect dialog opens a service on its sign-in, with the API key one click away", async () => {
    server.use(linearBlockAndSignIn);
    render(<Harness />);
    fireEvent.click(screen.getByText("open-connect"));

    const dialog = await screen.findByRole("dialog");
    await openLinear(dialog);
    expect(await within(dialog).findByText("Sign in to Linear.")).toBeDefined();
    expect(within(dialog).queryByPlaceholderText("My Linear key")).toBeNull();

    await userEvent.click(
      within(dialog).getByRole("button", {
        name: "More ways to connect",
      }),
    );
    expect(
      await within(dialog).findByPlaceholderText("My Linear key"),
    ).toBeDefined();
    expect(within(dialog).queryByText("Sign in to Linear.")).toBeNull();

    await userEvent.click(
      within(dialog).getByRole("button", {
        name: "Use the Linear sign-in instead",
      }),
    );
    expect(await within(dialog).findByText("Sign in to Linear.")).toBeDefined();
  });

  test("the integrations modal's connect dialog also opens on the sign-in", async () => {
    server.use(linearBlockAndSignIn);
    render(<Harness />);
    fireEvent.click(screen.getByText("open-integrations"));

    await userEvent.click(
      (await screen.findAllByRole("button", { name: /Connect Service/ }))[0],
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Connect a service",
    });
    await openLinear(dialog);
    expect(await within(dialog).findByText("Sign in to Linear.")).toBeDefined();
  });

  test("without the sign-in preference a service opens on its API key, with the sign-in one click away", async () => {
    server.use(linearBlockAndSignIn);
    render(<ConnectServiceDialog open onOpenChange={vi.fn()} />);

    const dialog = await screen.findByRole("dialog");
    await openLinear(dialog);
    expect(
      await within(dialog).findByPlaceholderText("My Linear key"),
    ).toBeDefined();
    expect(within(dialog).queryByText("Sign in to Linear.")).toBeNull();

    await userEvent.click(
      within(dialog).getByRole("button", {
        name: "Use the Linear sign-in instead",
      }),
    );
    expect(await within(dialog).findByText("Sign in to Linear.")).toBeDefined();
    expect(
      within(dialog).getByRole("button", {
        name: "More ways to connect",
      }),
    ).toBeDefined();
  });

  test("renders the connect dialog from a ?modal=connect deep link", async () => {
    render(
      <NuqsTestingAdapter searchParams="?modal=connect">
        <Harness />
      </NuqsTestingAdapter>,
    );

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Connect a service")).toBeDefined();
  });

  test("closing the connect dialog clears the modal query param", async () => {
    const onUrlUpdate = vi.fn();
    render(
      <NuqsTestingAdapter
        searchParams="?modal=connect"
        onUrlUpdate={onUrlUpdate}
      >
        <Harness />
      </NuqsTestingAdapter>,
    );

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));

    await vi.waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    await vi.waitFor(() => {
      expect(
        onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("modal"),
      ).toBeNull();
      expect(onUrlUpdate).toHaveBeenCalled();
    });
  });

  test("connecting from an expert chat grants the new credential to that expert", async () => {
    const grants = arrangeSentryGrantForMaria();

    render(<Harness />);
    fireEvent.click(screen.getByText("open-connect"));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      await within(dialog).findByRole("button", { name: /Sentry/ }),
    );
    const input = await within(dialog).findByLabelText("API token");
    fireEvent.change(input, { target: { value: "sntrys_token" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save token" }));

    await waitFor(() => expect(grants.granted).toEqual(["cred-sentry"]));
  });

  test("connecting from the integrations modal in an expert chat grants the new credential to that expert", async () => {
    const grants = arrangeSentryGrantForMaria();

    render(<Harness />);
    fireEvent.click(screen.getByText("open-integrations"));
    await userEvent.click(
      (await screen.findAllByRole("button", { name: /Connect Service/ }))[0],
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Connect a service",
    });
    fireEvent.click(
      await within(dialog).findByRole("button", { name: /Sentry/ }),
    );
    const input = await within(dialog).findByLabelText("API token");
    fireEvent.change(input, { target: { value: "sntrys_token" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save token" }));

    await waitFor(() => expect(grants.granted).toEqual(["cred-sentry"]));
  });
});

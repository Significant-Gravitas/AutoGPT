import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toastSpy(...args),
  useToast: () => ({ toast: toastSpy }),
  useToastOnFail: () => () => {},
}));

const openOAuthPopup = vi.fn();
vi.mock("@/lib/oauth-popup", () => ({
  OAUTH_ERROR_POPUP_BLOCKED: "Popup blocked",
  preOpenOAuthPopup: () => null,
  openOAuthPopup: (...args: unknown[]) => openOAuthPopup(...args),
}));

import { ExpertConnectServiceDialog } from "../ExpertConnectServiceDialog";

const PROVIDERS_URL =
  "http://localhost:3000/api/proxy/api/integrations/providers";
const CREDENTIALS_URL =
  "http://localhost:3000/api/proxy/api/integrations/credentials";
const CREATE_CREDENTIALS_URL =
  "http://localhost:3000/api/proxy/api/integrations/:provider/credentials";
const OAUTH_LOGIN_URL =
  "http://localhost:3000/api/proxy/api/integrations/:provider/login";
const OAUTH_CALLBACK_URL =
  "http://localhost:3000/api/proxy/api/integrations/:provider/callback";

const REGISTRY = [
  {
    name: "github",
    description: "Code host",
    supported_auth_types: ["oauth2", "api_key"],
  },
  {
    name: "smtp",
    description: "Mail relay",
    supported_auth_types: ["user_password"],
  },
  {
    name: "http",
    description: "Generic HTTP requests",
    supported_auth_types: ["host_scoped"],
  },
];

function stubDialog({ connectedProviders = [] as string[] } = {}) {
  server.use(
    http.get(PROVIDERS_URL, () => HttpResponse.json(REGISTRY)),
    http.get(CREDENTIALS_URL, () =>
      HttpResponse.json(
        connectedProviders.map((provider, index) => ({
          id: `cred-${index}`,
          provider,
          type: "api_key",
          title: `${provider} key`,
          scopes: null,
          username: null,
        })),
      ),
    ),
  );
}

function renderDialog({
  initialProviderId,
}: { initialProviderId?: string } = {}) {
  const onClose = vi.fn();
  const onConnected = vi.fn();
  render(
    <ExpertConnectServiceDialog
      open
      expertName="Maria"
      initialProviderId={initialProviderId}
      onClose={onClose}
      onConnected={onConnected}
    />,
  );
  return { onClose, onConnected };
}

async function openGithub() {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: /GitHub/ }));
  await screen.findByRole("heading", { name: "Connect AutoGPT to GitHub" });
  return user;
}

beforeEach(() => {
  toastSpy.mockClear();
  openOAuthPopup.mockReset();
});

describe("ExpertConnectServiceDialog — picking a provider", () => {
  it("offers an MCP preset through its own sign-in step", async () => {
    stubDialog();
    server.use(
      http.get(PROVIDERS_URL, () =>
        HttpResponse.json([
          ...REGISTRY,
          {
            name: "mcp_notion",
            display_name: "Notion",
            supported_auth_types: [],
            mcp_server: {
              server_url: "https://mcp.notion.com/mcp",
              documentation_url: "https://developers.notion.com/guides/mcp",
              setup_instructions: "Sign in to Notion.",
              connection_mode: "hosted",
              auth_methods: ["oauth"],
            },
          },
        ]),
      ),
    );
    renderDialog();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /Notion/ }));

    // The service's own sign-in, not the native-method picker.
    expect(
      await screen.findByRole("heading", {
        name: /^Connect (?!AutoGPT).*Notion/,
      }),
    ).toBeDefined();
    expect(screen.queryByText(/Choose how you'd like to connect/)).toBeNull();
    // The preset panel carries its own actions, so the footer goes away.
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Back" }));

    expect(await screen.findByLabelText("Search services")).toBeDefined();
  });

  it("opens the connect step with only the methods the provider supports", async () => {
    stubDialog();
    renderDialog();

    const user = await openGithub();

    expect(
      screen.getByText("Choose how you'd like to connect your GitHub account."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: /OAuth/ })).toBeDefined();
    expect(screen.getByRole("button", { name: /API Key/ })).toBeDefined();
    expect(
      screen.queryByRole("button", { name: /Username & password/ }),
    ).toBeNull();
    // Continue stays inert until a method is chosen.
    expect(screen.getByRole("button", { name: "Continue" })).toHaveProperty(
      "disabled",
      true,
    );

    await user.click(screen.getByRole("button", { name: "Back" }));

    expect(await screen.findByLabelText("Search services")).toBeDefined();
    expect(
      screen.queryByRole("heading", { name: "Connect AutoGPT to GitHub" }),
    ).toBeNull();
  });

  it("expands the user/password form in place, with no footer Continue", async () => {
    stubDialog();
    renderDialog();

    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /Smtp/ }));
    await screen.findByRole("heading", { name: "Connect AutoGPT to Smtp" });

    await user.click(
      screen.getByRole("button", { name: /Username & password/ }),
    );

    expect(await screen.findByLabelText("Username")).toBeDefined();
    expect(
      screen.getByLabelText("Password", { selector: "input" }),
    ).toBeDefined();
    expect(screen.queryByText("No connection method available")).toBeNull();
    // The card submits itself, so the footer's Continue stays away.
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();
  });

  it("opens straight on the connect step for a provider picked elsewhere", async () => {
    stubDialog();
    renderDialog({ initialProviderId: "github" });

    expect(
      await screen.findByRole("heading", { name: "Connect AutoGPT to GitHub" }),
    ).toBeDefined();
    expect(screen.queryByLabelText("Search services")).toBeNull();
  });

  it("closes on Escape from the connect step", async () => {
    stubDialog();
    const { onClose } = renderDialog();
    const user = await openGithub();

    await user.keyboard("{Escape}");

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  it("marks providers that already hold credentials as connected", async () => {
    stubDialog({ connectedProviders: ["github"] });
    renderDialog();

    const github = await screen.findByRole("button", { name: /GitHub/ });
    const smtp = screen.getByRole("button", { name: /Smtp/ });

    // The connected mark is a decorative icon with no accessible name, so
    // it can only be identified by its tint.
    expect(github.querySelector(".text-emerald-500")).not.toBeNull();
    expect(smtp.querySelector(".text-emerald-500")).toBeNull();
  });
});

describe("ExpertConnectServiceDialog — inline host-scoped flow", () => {
  async function openHttpHostCard() {
    stubDialog();
    const { onConnected } = renderDialog();

    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /Http/ }));
    await screen.findByRole("heading", { name: "Connect AutoGPT to Http" });
    await user.click(screen.getByRole("button", { name: /^Website access/ }));
    return { onConnected, user };
  }

  it("expands the host form in place instead of the unsupported notice", async () => {
    const { user } = await openHttpHostCard();

    expect(await screen.findByLabelText("Host")).toBeDefined();
    expect(screen.getByLabelText("Header name")).toBeDefined();
    expect(screen.getByLabelText("Header value")).toBeDefined();
    expect(screen.queryByText("No connection method available")).toBeNull();
    // No block is in scope here, so the field says what to type rather than
    // presenting an unexplained empty box.
    expect(
      screen.getByText("The host of the URL this block will call."),
    ).toBeDefined();
    expect(screen.getByLabelText("Host")).toHaveProperty("readOnly", false);
    // The card submits itself, so the footer's Continue stays away.
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();

    expect(screen.getByRole("button", { name: "Connect" })).toHaveProperty(
      "disabled",
      true,
    );

    // A host with no headers saves a credential that adds nothing to a
    // request, so the host alone must not be enough.
    await user.type(screen.getByLabelText("Host"), "api.example.com");
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Connect" })).toHaveProperty(
        "disabled",
        true,
      ),
    );

    await user.type(screen.getByLabelText("Header name"), "Authorization");
    await user.type(screen.getByLabelText("Header value"), "<test-token>");

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Connect" })).toHaveProperty(
        "disabled",
        false,
      ),
    );
  });

  it("rejects a full URL where the host belongs", async () => {
    const { user } = await openHttpHostCard();

    await user.type(
      await screen.findByLabelText("Host"),
      "https://api.example.com",
    );
    await user.type(screen.getByLabelText("Header name"), "Authorization");
    await user.type(screen.getByLabelText("Header value"), "<test-token>");

    expect(
      await screen.findByText(
        "Enter only the host (e.g. api.example.com), not a full URL",
      ),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Connect" })).toHaveProperty(
      "disabled",
      true,
    );
  });

  it("posts the host and its headers to the picked provider", async () => {
    const requests: { provider: string; body: unknown }[] = [];
    server.use(
      http.post(CREATE_CREDENTIALS_URL, async ({ params, request }) => {
        requests.push({
          provider: String(params.provider),
          body: await request.json(),
        });
        return HttpResponse.json(
          {
            id: "cred-http",
            provider: "http",
            type: "host_scoped",
            title: "api.example.com",
            scopes: null,
            username: null,
          },
          { status: 201 },
        );
      }),
    );

    const { onConnected, user } = await openHttpHostCard();
    await user.type(await screen.findByLabelText("Host"), "api.example.com");
    await user.type(screen.getByLabelText("Header name"), "Authorization");
    await user.type(screen.getByLabelText("Header value"), "<test-token>");

    await user.click(screen.getByRole("button", { name: "Connect" }));

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0].provider).toBe("http");
    expect(requests[0].body).toMatchObject({
      provider: "http",
      type: "host_scoped",
      title: "api.example.com",
      host: "api.example.com",
      headers: { Authorization: "<test-token>" },
    });
    expect(toastSpy).toHaveBeenCalledWith({
      title: "Host credentials saved",
      variant: "success",
    });
    await waitFor(() =>
      expect(onConnected).toHaveBeenCalledWith(
        expect.objectContaining({ id: "cred-http" }),
      ),
    );
  });
});

describe("ExpertConnectServiceDialog — inline user/password flow", () => {
  it("posts the username and password to the picked provider", async () => {
    stubDialog();
    const requests: { provider: string; body: unknown }[] = [];
    server.use(
      http.post(CREATE_CREDENTIALS_URL, async ({ params, request }) => {
        requests.push({
          provider: String(params.provider),
          body: await request.json(),
        });
        return HttpResponse.json(
          {
            id: "cred-smtp",
            provider: "smtp",
            type: "user_password",
            title: "Mail relay",
            scopes: null,
            username: "postmaster",
          },
          { status: 201 },
        );
      }),
    );

    const { onConnected } = renderDialog();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /Smtp/ }));
    await screen.findByRole("heading", { name: "Connect AutoGPT to Smtp" });
    await user.click(
      screen.getByRole("button", { name: /Username & password/ }),
    );

    await user.type(await screen.findByLabelText("Name"), "Mail relay");
    await user.type(screen.getByLabelText("Username"), "postmaster");
    await user.type(
      screen.getByLabelText("Password", { selector: "input" }),
      "<test-password>",
    );

    await user.click(screen.getByRole("button", { name: "Connect" }));

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0].provider).toBe("smtp");
    expect(requests[0].body).toMatchObject({
      provider: "smtp",
      type: "user_password",
      title: "Mail relay",
      username: "postmaster",
      password: "<test-password>",
    });
    await waitFor(() =>
      expect(onConnected).toHaveBeenCalledWith(
        expect.objectContaining({ id: "cred-smtp" }),
      ),
    );
  });
});

describe("ExpertConnectServiceDialog — inline API key flow", () => {
  it("expands the key form in place and gates Continue on a valid form", async () => {
    stubDialog();
    renderDialog();
    const user = await openGithub();

    await user.click(screen.getByRole("button", { name: /API Key/ }));

    const apiKeyField = await screen.findByLabelText("API key", {
      selector: "input",
    });
    expect(screen.getByLabelText("Name")).toBeDefined();
    expect(screen.getByLabelText("Expires (optional)")).toBeDefined();
    expect(screen.getByRole("button", { name: "Continue" })).toHaveProperty(
      "disabled",
      true,
    );

    await user.type(screen.getByLabelText("Name"), "Work key");
    await user.type(apiKeyField, "<test-api-key>");

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Continue" })).toHaveProperty(
        "disabled",
        false,
      ),
    );
  });

  it("posts the key, hands the credential back and slides back to the list", async () => {
    stubDialog();
    const requests: { provider: string; body: unknown }[] = [];
    server.use(
      http.post(CREATE_CREDENTIALS_URL, async ({ params, request }) => {
        requests.push({
          provider: String(params.provider),
          body: await request.json(),
        });
        return HttpResponse.json(
          {
            id: "cred-new",
            provider: "github",
            type: "api_key",
            title: "Work key",
            scopes: null,
            username: null,
          },
          { status: 201 },
        );
      }),
    );

    const { onConnected } = renderDialog();
    const user = await openGithub();
    await user.click(screen.getByRole("button", { name: /API Key/ }));
    await user.type(await screen.findByLabelText("Name"), "Work key");
    await user.type(
      screen.getByLabelText("API key", { selector: "input" }),
      "<test-api-key>",
    );

    await user.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0].provider).toBe("github");
    expect(requests[0].body).toMatchObject({
      provider: "github",
      type: "api_key",
      title: "Work key",
      api_key: "<test-api-key>",
    });
    expect(toastSpy).toHaveBeenCalledWith({
      title: "API key saved",
      variant: "success",
    });
    expect(onConnected).toHaveBeenCalledWith(
      expect.objectContaining({ id: "cred-new" }),
    );

    // Success returns to the list so more tools can be wired up.
    expect(await screen.findByLabelText("Search services")).toBeDefined();
    expect(
      screen.queryByRole("heading", { name: "Connect AutoGPT to GitHub" }),
    ).toBeNull();
  });

  it("keeps the form open and reports the failure when the key is rejected", async () => {
    stubDialog();
    server.use(
      http.post(CREATE_CREDENTIALS_URL, () =>
        HttpResponse.json({ detail: "Invalid API key" }, { status: 400 }),
      ),
    );

    const { onConnected } = renderDialog();
    const user = await openGithub();
    await user.click(screen.getByRole("button", { name: /API Key/ }));
    await user.type(await screen.findByLabelText("Name"), "Work key");
    await user.type(
      screen.getByLabelText("API key", { selector: "input" }),
      "nope",
    );

    await user.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't save API key",
          variant: "destructive",
        }),
      ),
    );
    expect(
      screen.getByRole("heading", { name: "Connect AutoGPT to GitHub" }),
    ).toBeDefined();
    expect(onConnected).not.toHaveBeenCalled();
  });
});

describe("ExpertConnectServiceDialog — without a plan", () => {
  it("points to the plans instead of reporting a failed save", async () => {
    stubDialog();
    server.use(
      http.post(CREATE_CREDENTIALS_URL, () =>
        HttpResponse.json(
          { detail: "A subscription is required." },
          { status: 402 },
        ),
      ),
    );

    renderDialog();
    const user = await openGithub();
    await user.click(screen.getByRole("button", { name: /API Key/ }));
    await user.type(await screen.findByLabelText("Name"), "Work key");
    await user.type(
      screen.getByLabelText("API key", { selector: "input" }),
      "<test-api-key>",
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Connecting a service needs an active plan",
        }),
      ),
    );
    expect(toastSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "Couldn't save API key" }),
    );
  });
});

describe("ExpertConnectServiceDialog — OAuth without a plan", () => {
  it("points to the plans before any sign-in window opens", async () => {
    stubDialog();
    server.use(
      http.get(OAUTH_LOGIN_URL, () =>
        HttpResponse.json(
          { detail: "A subscription is required." },
          { status: 402 },
        ),
      ),
    );

    renderDialog();
    const user = await openGithub();
    await user.click(screen.getByRole("button", { name: /OAuth/ }));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Connecting a service needs an active plan",
        }),
      ),
    );
    expect(openOAuthPopup).not.toHaveBeenCalled();
    expect(toastSpy).not.toHaveBeenCalledWith(
      expect.objectContaining({ title: "OAuth connection failed" }),
    );
  });
});

describe("ExpertConnectServiceDialog — inline OAuth flow", () => {
  it("runs the whole OAuth exchange without leaving the dialog", async () => {
    stubDialog();
    const exchanges: { provider: string; body: unknown }[] = [];
    let loginUrlRequests = 0;
    server.use(
      http.get(OAUTH_LOGIN_URL, () => {
        loginUrlRequests++;
        return HttpResponse.json({
          login_url: "https://github.test/oauth",
          state_token: "state-abc",
        });
      }),
      http.post(OAUTH_CALLBACK_URL, async ({ params, request }) => {
        exchanges.push({
          provider: String(params.provider),
          body: await request.json(),
        });
        return HttpResponse.json({
          id: "cred-oauth",
          provider: "github",
          type: "oauth2",
          title: "GitHub",
          scopes: [],
          username: "octocat",
        });
      }),
    );
    openOAuthPopup.mockReturnValue({
      promise: Promise.resolve({ code: "auth-code", state: "state-abc" }),
      cleanup: { abort: vi.fn() },
      popupBlocked: false,
      fallbackBlocked: false,
    });

    const { onConnected } = renderDialog();
    const user = await openGithub();

    await user.click(screen.getByRole("button", { name: /OAuth/ }));
    // OAuth has no inline inputs — it is driven entirely by Continue.
    expect(
      screen.queryByLabelText("API key", { selector: "input" }),
    ).toBeNull();

    await user.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => expect(exchanges).toHaveLength(1));
    expect(loginUrlRequests).toBe(1);
    expect(openOAuthPopup).toHaveBeenCalledWith(
      "https://github.test/oauth",
      expect.objectContaining({ stateToken: "state-abc" }),
    );
    expect(exchanges[0].provider).toBe("github");
    expect(exchanges[0].body).toMatchObject({
      code: "auth-code",
      state_token: "state-abc",
    });
    expect(toastSpy).toHaveBeenCalledWith({
      title: "Connected via OAuth",
      variant: "success",
    });
    expect(onConnected).toHaveBeenCalledWith(
      expect.objectContaining({ id: "cred-oauth" }),
    );
    expect(await screen.findByLabelText("Search services")).toBeDefined();
  });
});

describe("ExpertConnectServiceDialog — search and failure states", () => {
  it("searches the whole registry and reports when nothing matches", async () => {
    stubDialog();
    renderDialog();

    const user = userEvent.setup();
    const search = await screen.findByLabelText("Search services");

    await user.type(search, "github");

    // The query is debounced, so the unmatched row is dropped a beat later.
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: /Smtp/ })).toBeNull(),
    );
    expect(screen.getByRole("button", { name: /GitHub/ })).toBeDefined();

    await user.clear(search);
    await user.type(search, "zzz");

    expect(await screen.findByText('No services match "zzz"')).toBeDefined();
  });

  it("surfaces a retryable error when the provider list cannot be loaded", async () => {
    let attempts = 0;
    server.use(
      http.get(PROVIDERS_URL, () => {
        attempts++;
        if (attempts === 1) {
          return HttpResponse.json({ detail: "boom" }, { status: 500 });
        }
        return HttpResponse.json(REGISTRY);
      }),
      http.get(CREDENTIALS_URL, () => HttpResponse.json([])),
    );

    renderDialog();

    const errorCard = await screen.findByText(
      "We had the following error when retrieving services:",
    );
    expect(errorCard).toBeDefined();
    expect(screen.queryByLabelText("Search services")).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: /Try Again/ }));

    expect(await screen.findByLabelText("Search services")).toBeDefined();
    expect(attempts).toBeGreaterThan(1);
  });
});

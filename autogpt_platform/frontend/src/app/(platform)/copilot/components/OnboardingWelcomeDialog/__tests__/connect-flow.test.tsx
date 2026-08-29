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

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toastSpy(...args),
  useToast: () => ({ toast: toastSpy }),
  useToastOnFail: () => () => {},
}));

// The recommended list — the panel's default view — is flag-gated.
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === actual.Flag.ONBOARDING_BRAIN_DUMP,
  };
});

const openOAuthPopup = vi.fn();
vi.mock("@/lib/oauth-popup", () => ({
  OAUTH_ERROR_POPUP_BLOCKED: "Popup blocked",
  preOpenOAuthPopup: () => null,
  openOAuthPopup: (...args: unknown[]) => openOAuthPopup(...args),
}));

import { ConnectToolsPanel } from "../ConnectToolsPanel";

const PROVIDERS_URL =
  "http://localhost:3000/api/proxy/api/integrations/providers";
const CREDENTIALS_URL =
  "http://localhost:3000/api/proxy/api/integrations/credentials";
const RECOMMENDED_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/recommended-providers";
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
];

function stubPanel({
  connectedProviders = [] as string[],
  recommended = ["github", "smtp"],
} = {}) {
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
    http.get(RECOMMENDED_URL, () =>
      HttpResponse.json({
        ready: true,
        providers: recommended.map((provider) => ({
          provider,
          reason: `Because you mentioned ${provider}`,
        })),
      }),
    ),
  );
}

function renderPanel() {
  return render(<ConnectToolsPanel onBack={vi.fn()} onNext={vi.fn()} />);
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

describe("ConnectToolsPanel — picking a provider", () => {
  it("opens the connect step with only the methods the provider supports", async () => {
    stubPanel();
    renderPanel();

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

  it("offers no Continue for a method the dialog cannot drive", async () => {
    stubPanel();
    renderPanel();

    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /Smtp/ }));
    await screen.findByRole("heading", { name: "Connect AutoGPT to Smtp" });

    await user.click(
      screen.getByRole("button", { name: /Username & password/ }),
    );

    expect(
      await screen.findByText("No connection method available"),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();
  });

  it("returns to the list on Escape rather than leaving the picker", async () => {
    stubPanel();
    const onBack = vi.fn();
    render(<ConnectToolsPanel onBack={onBack} onNext={vi.fn()} />);
    const user = await openGithub();

    await user.keyboard("{Escape}");

    expect(await screen.findByLabelText("Search services")).toBeDefined();
    expect(onBack).not.toHaveBeenCalled();

    // Only from the list does Escape hand back to the dialog.
    await user.keyboard("{Escape}");

    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it("marks providers that already hold credentials as connected", async () => {
    stubPanel({ connectedProviders: ["github"] });
    renderPanel();

    const github = await screen.findByRole("button", { name: /GitHub/ });
    const smtp = screen.getByRole("button", { name: /Smtp/ });

    // The connected mark is a decorative icon with no accessible name, so
    // it can only be identified by its tint.
    expect(github.querySelector(".text-emerald-500")).not.toBeNull();
    expect(smtp.querySelector(".text-emerald-500")).toBeNull();
  });
});

describe("ConnectToolsPanel — inline API key flow", () => {
  it("expands the key form in place and gates Continue on a valid form", async () => {
    stubPanel();
    renderPanel();
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
    await user.type(apiKeyField, "sk-live-123");

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Continue" })).toHaveProperty(
        "disabled",
        false,
      ),
    );
  });

  it("posts the key to the picked provider and slides back to the list", async () => {
    stubPanel();
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

    renderPanel();
    const user = await openGithub();
    await user.click(screen.getByRole("button", { name: /API Key/ }));
    await user.type(await screen.findByLabelText("Name"), "Work key");
    await user.type(
      screen.getByLabelText("API key", { selector: "input" }),
      "sk-live-123",
    );

    await user.click(screen.getByRole("button", { name: "Continue" }));

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0].provider).toBe("github");
    expect(requests[0].body).toMatchObject({
      provider: "github",
      type: "api_key",
      title: "Work key",
      api_key: "sk-live-123",
    });
    expect(toastSpy).toHaveBeenCalledWith({
      title: "API key saved",
      variant: "success",
    });

    // Success returns to the list so more tools can be wired up.
    expect(await screen.findByLabelText("Search services")).toBeDefined();
    expect(
      screen.queryByRole("heading", { name: "Connect AutoGPT to GitHub" }),
    ).toBeNull();
  });

  it("keeps the form open and reports the failure when the key is rejected", async () => {
    stubPanel();
    server.use(
      http.post(CREATE_CREDENTIALS_URL, () =>
        HttpResponse.json({ detail: "Invalid API key" }, { status: 400 }),
      ),
    );

    renderPanel();
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
  });
});

describe("ConnectToolsPanel — inline OAuth flow", () => {
  it("runs the whole OAuth exchange without leaving the panel", async () => {
    stubPanel();
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

    renderPanel();
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
    expect(await screen.findByLabelText("Search services")).toBeDefined();
  });
});

describe("ConnectToolsPanel — search and failure states", () => {
  it("searches the whole registry and reports when nothing matches", async () => {
    stubPanel({ recommended: [] });
    renderPanel();

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
      http.get(RECOMMENDED_URL, () =>
        HttpResponse.json({ ready: true, providers: [] }),
      ),
    );

    renderPanel();

    const errorCard = await screen.findByText(
      "We had the following error when retrieving providers:",
    );
    expect(errorCard).toBeDefined();
    expect(screen.queryByLabelText("Search services")).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: /Try Again/ }));

    expect(await screen.findByLabelText("Search services")).toBeDefined();
    expect(attempts).toBeGreaterThan(1);
  });
});

describe("ConnectProviderRow — missing artwork", () => {
  it("falls back to the provider initial when the logo fails to load", async () => {
    stubPanel({ recommended: ["github"] });
    renderPanel();

    const row = await screen.findByRole("button", { name: /GitHub/ });
    const logo = row.querySelector("img");
    expect(logo).not.toBeNull();

    await waitFor(() => {
      logo?.dispatchEvent(new Event("error"));
      expect(within(row).getByText("G")).toBeDefined();
    });
    expect(row.querySelector("img")).toBeNull();
  });
});

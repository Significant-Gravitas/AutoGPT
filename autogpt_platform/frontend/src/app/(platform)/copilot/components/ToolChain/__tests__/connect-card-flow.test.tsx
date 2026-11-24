/**
 * End-to-end regression for the copilot's Connect card.
 *
 * Every layer of this path had tests and every layer passed, yet composing
 * them produced a card that could never be completed. So this test drives the
 * whole path, mocking only its two ends:
 *
 *     ToolChain → SetupRequirementsCard → ChainActionCard → ConnectorRow
 *               → ConnectCredentialDialog → useOAuthConnect → the API
 *
 * The API is MSW rather than a live backend, and the consent screen is the one
 * unavoidable fake — no CI job can have a human approve a real OAuth grant.
 * Every component in between is the real one.
 *
 * The reload and mid-turn tests start one layer up, at the transcript,
 * because whether a card is in the newest answer is decided there:
 *
 *     ChatMessagesContainer → MessageRow → ChainMessageParts → ToolChain → …
 *
 * Only the transcript's scroll container and the chrome around its rows are
 * stubbed for them.
 */

import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import userEvent from "@testing-library/user-event";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { HttpResponse, http } from "msw";
import type { ReactNode } from "react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type MockInstance,
} from "vitest";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useConnectedProvidersStore } from "@/app/(platform)/copilot/connectedProvidersStore";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChatMessagesContainer } from "../../ChatMessagesContainer/ChatMessagesContainer";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { ToolChain } from "../ToolChain";
import { PendingQuestionsContext } from "../../QuestionDock/PendingQuestionsContext";
import type { PendingQuestions } from "../../QuestionDock/helpers";

// CredentialsProvider only fetches for a signed-in user; the shared setup
// signs everyone out.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, isUserLoading: false, user: null }),
}));

// The popup is the consent screen; nothing else about the flow is faked.
// It resolves with the code the real callback handler would receive.
vi.mock("@/lib/oauth-popup", () => ({
  OAUTH_ERROR_POPUP_BLOCKED: "popup blocked",
  preOpenOAuthPopup: () => null,
  openOAuthPopup: () => ({
    promise: Promise.resolve({ code: "oauth-code", state: "state-token" }),
    cleanup: { abort: () => undefined },
    popupBlocked: false,
    fallbackBlocked: false,
  }),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn() }),
  useToastOnFail: () => () => () => {},
}));

// The transcript's scroll container and the chrome around its rows, which the
// reload tests do not exercise.
vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => ({
    scrollRef: {
      current: { scrollHeight: 100, scrollTop: 0, clientHeight: 500 },
    },
  }),
}));
vi.mock("@/components/organisms/Conversation/Conversation", () => ({
  Conversation: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationContent: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationScrollButton: () => null,
}));
vi.mock(
  "../../ChatMessagesContainer/components/AssistantMessageActions",
  () => ({ AssistantMessageActions: () => null }),
);
vi.mock("../../JobStatsBar/TurnStatsBar", () => ({ TurnStatsBar: () => null }));
vi.mock("../../CopilotPendingReviews/CopilotPendingReviews", () => ({
  CopilotPendingReviews: () => null,
}));
vi.mock("../../ChatMinimap/ChatMinimap", () => ({ ChatMinimap: () => null }));
vi.mock("../../DrawnLineLoader/DrawnLineLoader", () => ({
  DrawnLineLoader: () => null,
}));

import { toast } from "@/components/molecules/Toast/use-toast";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

const REQUIRED_SCOPE = "repo";
const SESSION_ID = "session-1";

// The scopes the browser actually asked the consent screen for. `null` means
// the login endpoint was never called; `""` means it was called without them.
let requestedScopes: string | null = null;
// The account the sign-in asked to upgrade, if any.
let upgradedCredentialID: string | null = null;
let savedCredentials: Record<string, unknown>[] = [];

describe("copilot Connect card", () => {
  beforeEach(() => {
    requestedScopes = null;
    upgradedCredentialID = null;
    // The user already has GitHub connected, but without the scope this block
    // needs — the exact state in which the card is surfaced.
    savedCredentials = [
      {
        id: "cred-old",
        provider: "github",
        type: "oauth2",
        title: "GitHub",
        scopes: ["notifications", "read:user"],
      },
    ];
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "github",
            description: "Issues, pull requests, repositories",
          },
        ]),
      ),
      http.get("*/api/integrations/providers/system", () =>
        HttpResponse.json([]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(savedCredentials),
      ),
      http.get("*/api/integrations/github/login", ({ request }) => {
        const query = new URL(request.url).searchParams;
        requestedScopes = query.get("scopes") ?? "";
        upgradedCredentialID = query.get("credential_id");
        return HttpResponse.json({
          login_url: "https://github.com/login/oauth/authorize",
          state_token: "state-token",
        });
      }),
      // The real backend grants what the login URL asked for, so the scopes
      // the frontend dropped are the scopes the new credential lacks.
      http.post("*/api/integrations/github/callback", () => {
        const granted = requestedScopes ? requestedScopes.split(",") : [];
        savedCredentials = [
          ...savedCredentials,
          {
            id: "cred-new",
            provider: "github",
            type: "oauth2",
            title: "GitHub",
            scopes: granted,
          },
        ];
        return HttpResponse.json(savedCredentials.at(-1));
      }),
    );
  });

  afterEach(resetStores);

  it("asks the consent screen for the scopes the block requires", async () => {
    renderChain();

    await completeConnectFlow();

    // Pre-fix this was "" — the chain initiated OAuth with no scopes at all,
    // so the granted credential could never satisfy the card that asked for it.
    await waitFor(() => expect(requestedScopes).toBe(REQUIRED_SCOPE));
  });

  it("marks the row connected and continues the chat once connected", async () => {
    const { onSend } = renderChain();

    await completeConnectFlow();

    // The chain renders no Proceed for a connectors-only card, so without an
    // automatic follow-up turn the chat stalls with the credential connected.
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toContain(
      "I've configured the required credentials.",
    );
    await screen.findByText("Connected. Continuing…");
  });

  it("offers an API key below sign-in when either method can satisfy the workflow", async () => {
    server.use(
      http.post(
        "*/api/integrations/github/credentials",
        async ({ request }) => {
          const body = (await request.json()) as Record<string, unknown>;
          const stored = {
            id: "github-key",
            provider: "github",
            type: "api_key",
            title: body.title,
          };
          savedCredentials = [stored];
          return HttpResponse.json(stored);
        },
      ),
    );
    const { onSend } = renderChain([
      setupRequirementsPart(["oauth2", "api_key"]),
    ]);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Connect" }));
    await screen.findByRole("button", { name: "Continue with GitHub" });
    await user.click(screen.getByText("Use an API key instead"));
    await user.type(await screen.findByLabelText("Name"), "Work account");
    await user.type(screen.getByLabelText("API key"), "test-github-key");
    await user.click(screen.getByRole("button", { name: "Save and connect" }));
    await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
    expect(requestedScopes).toBeNull();
  });

  it("offers the existing account for upgrade instead of a fresh grant", async () => {
    renderChain();

    await completeConnectFlow();

    // Without this the backend cannot union scopes, and a narrower grant is
    // stored as a second row that no ConnectorRow can ever resolve.
    await waitFor(() => expect(upgradedCredentialID).toBe("cred-old"));
  });

  it("leaves the row unconnected when the grant comes back short", async () => {
    // A provider that silently downgrades the grant must not read as connected
    // — that was the shape of the original "Connected, then 401" report.
    server.use(
      http.get("*/api/integrations/github/login", () => {
        requestedScopes = "";
        return HttpResponse.json({
          login_url: "https://github.com/login/oauth/authorize",
          state_token: "state-token",
        });
      }),
    );
    const { onSend } = renderChain();

    await completeConnectFlow();

    await waitFor(() => expect(savedCredentials).toHaveLength(2));
    await screen.findByRole("button", { name: "Connect" });
    expect(onSend).not.toHaveBeenCalled();
  });
});

describe("copilot Connect card, cards that owe more than a credential", () => {
  beforeEach(() => {
    requestedScopes = null;
    upgradedCredentialID = null;
    savedCredentials = [];
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([{ name: "github", description: "Repositories" }]),
      ),
      http.get("*/api/integrations/providers/system", () =>
        HttpResponse.json([]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(savedCredentials),
      ),
      http.get("*/api/integrations/github/login", ({ request }) => {
        requestedScopes = new URL(request.url).searchParams.get("scopes") ?? "";
        return HttpResponse.json({
          login_url: "https://github.com/login/oauth/authorize",
          state_token: "state-token",
        });
      }),
      http.post("*/api/integrations/github/callback", () => {
        savedCredentials = [oauthCredential("cred-new", [REQUIRED_SCOPE])];
        return HttpResponse.json(savedCredentials[0]);
      }),
    );
  });

  afterEach(resetStores);

  it("waits for the user when the card also collects run inputs", async () => {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[partWithInputs()]} isStreaming={false} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await completeConnectFlow();

    // Auto-sending here would run the block with the inputs the user has not
    // filled in yet, discarding them. Wait on the row resolving, not on the
    // MSW handler: `savedCredentials` is set inside the handler, before the
    // query invalidates or the auto-send effect could run.
    await screen.findByText("Connected");
    expect(onSend).not.toHaveBeenCalled();
  });

  it("gives a trigger card a Proceed instead of auto-sending it", async () => {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[triggerPart()]} isStreaming={false} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await completeConnectFlow();

    // buildTriggerSetupMessage carries the credential the webhook registers
    // under, so the account must be chosen rather than auto-picked — but
    // excluding the auto-send without adding a button leaves the card with no
    // way forward at all, which is worse than sending the wrong account.
    // The Proceed button is the positive signal that the card has finished
    // reacting, so it has to come before the negative assertion.
    await screen.findByRole("button", { name: "Proceed" });
    expect(onSend).not.toHaveBeenCalled();
  });

  it("offers no Proceed for a connected MCP row, which builds no message", async () => {
    savedCredentials = [
      {
        id: "cred-mcp",
        provider: "mcp",
        type: "oauth2",
        title: "Notion",
        host: "https://mcp.notion.com/mcp",
        scopes: [],
      },
    ];
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[mcpPart()]} isStreaming={false} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await screen.findByText(/Connect your apps|Your apps are connected/);
    // The MCP entry's buildMessage returns null, so a Proceed here would draft
    // an empty message and do nothing at all.
    expect(screen.queryByRole("button", { name: "Proceed" })).toBeNull();
  });

  it("gives a Proceed to a card whose rows an earlier card already satisfied", async () => {
    // connect_integration returns a card whether or not the user is already
    // connected, so this lands whenever the assistant offers a provider they
    // have. There is no sign-in to trigger the send and nothing to click.
    savedCredentials = [oauthCredential("cred-existing", [REQUIRED_SCOPE])];
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: SESSION_ID, providers: ["github"] });

    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[setupRequirementsPart()]} isStreaming={false} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await screen.findByRole("button", { name: "Proceed" });
    expect(onSend).not.toHaveBeenCalled();
  });

  it("gives a trigger card a Proceed when a sibling already connected its provider", async () => {
    // The auto-dismiss path fires before any account is selected and sends
    // `credentials={}`; the backend then re-issues the card, the re-issued one
    // has already lost the dedupe claim, and the chat waits forever.
    savedCredentials = [oauthCredential("cred-existing", [REQUIRED_SCOPE])];
    useConnectedProvidersStore
      .getState()
      .markConnected({ sessionID: SESSION_ID, providers: ["github"] });

    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[triggerPart()]} isStreaming={false} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await screen.findByRole("button", { name: "Proceed" });
    expect(onSend).not.toHaveBeenCalled();
  });
});

describe("copilot Connect card, a row whose schema widens mid-stream", () => {
  beforeEach(() => {
    requestedScopes = null;
    upgradedCredentialID = null;
    // Satisfies the first card's `repo` but not the second card's `workflow`.
    savedCredentials = [oauthCredential("cred-repo", [REQUIRED_SCOPE])];
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([{ name: "github", description: "Repositories" }]),
      ),
      http.get("*/api/integrations/providers/system", () =>
        HttpResponse.json([]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(savedCredentials),
      ),
    );
  });

  afterEach(resetStores);

  it("drops a selection the widened schema no longer satisfies", async () => {
    const onSend = vi.fn();
    const { rerender } = render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[setupRequirementsPart()]} isStreaming={true} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );
    await screen.findByText("Connected");

    rerender(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[setupRequirementsPart(), widerScopePart()]}
            isStreaming={false}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    // Cards stream in one commit at a time. Leaving the stale selection in
    // place renders "Connected" and lets Proceed send a credential missing
    // the scope the second card asked for.
    await screen.findByRole("button", { name: "Connect" });
    expect(onSend).not.toHaveBeenCalled();
  });
});

describe("copilot Connect card, upgrade-target selection", () => {
  beforeEach(() => {
    requestedScopes = null;
    upgradedCredentialID = null;
    savedCredentials = [];
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([{ name: "github", description: "Repositories" }]),
      ),
      http.get("*/api/integrations/providers/system", () =>
        HttpResponse.json([]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(savedCredentials),
      ),
      http.get("*/api/integrations/github/login", ({ request }) => {
        const query = new URL(request.url).searchParams;
        requestedScopes = query.get("scopes") ?? "";
        upgradedCredentialID = query.get("credential_id");
        return HttpResponse.json({
          login_url: "https://github.com/login/oauth/authorize",
          state_token: "state-token",
        });
      }),
      http.post("*/api/integrations/github/callback", () =>
        HttpResponse.json({ id: "cred-new", provider: "github" }),
      ),
    );
  });

  afterEach(resetStores);

  it("asks which account to update when two could be the one", async () => {
    savedCredentials = [
      oauthCredential("cred-a", ["notifications"]),
      oauthCredential("cred-b", ["read:user"]),
    ];

    renderChain();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Connect" }));

    // Choosing between them on the user's behalf would be a guess, and the
    // backend upgrades in place, so a wrong guess silently broadens the wrong
    // account. Signing in without a target stored a third credential instead.
    // So the user names the account.
    expect(await screen.findByText("Update a GitHub account")).toBeDefined();
    expect(requestedScopes).toBeNull();

    const accounts = await screen.findAllByRole("radio");
    await user.click(accounts[1]);
    await user.click(
      screen.getByRole("button", { name: "Update this account" }),
    );
    await user.click(
      await screen.findByRole("button", { name: "Continue with GitHub" }),
    );

    await waitFor(() => expect(upgradedCredentialID).toBe("cred-b"));
  });

  it("adds a new account beside two existing ones when asked to", async () => {
    savedCredentials = [
      oauthCredential("cred-a", ["notifications"]),
      oauthCredential("cred-b", ["read:user"]),
    ];

    renderChain();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Connect" }));
    await user.click(await screen.findByRole("button", { name: "Add new" }));
    await user.click(
      await screen.findByRole("button", { name: "Continue with GitHub" }),
    );

    await waitFor(() => expect(requestedScopes).toBe(REQUIRED_SCOPE));
    expect(upgradedCredentialID).toBeNull();
  });

  it.each([true, false])(
    "switches from a selected account to the new account (reported: %s)",
    async (reported) => {
      savedCredentials = [oauthCredential("cred-existing", [REQUIRED_SCOPE])];
      const selections: Record<string, string>[] = [];
      server.use(
        http.put(
          "*/api/chat/sessions/*/credential-selection",
          async ({ request }) => {
            const body = (await request.json()) as {
              selections: Record<string, string>;
            };
            selections.push(body.selections);
            return HttpResponse.json({ status: "ok" });
          },
        ),
        http.post("*/api/integrations/github/callback", () => {
          const added = oauthCredential("cred-new", [REQUIRED_SCOPE]);
          savedCredentials = [...savedCredentials, added];
          return HttpResponse.json(reported ? added : {});
        }),
      );
      const { onSend } = renderChain();
      const user = userEvent.setup();
      await user.click(
        await screen.findByRole("button", { name: "Change account" }),
      );
      await user.click(await screen.findByRole("button", { name: "Add new" }));
      await user.click(
        await screen.findByRole("button", { name: "Continue with GitHub" }),
      );
      await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
      expect(selections).toEqual([{ github: "cred-new" }]);
      expect(upgradedCredentialID).toBeNull();
    },
  );

  it.each(["stale", "failed"])(
    "continues with the newly connected account after a %s credential reload",
    async (reload) => {
      const original = oauthCredential("cred-existing", [REQUIRED_SCOPE]);
      savedCredentials = [original];
      let callbackCompleted = false;
      const selections: Record<string, string>[] = [];
      server.use(
        http.get("*/api/integrations/credentials", () => {
          if (callbackCompleted && reload === "failed") {
            return HttpResponse.json(
              { detail: "List unavailable" },
              { status: 500 },
            );
          }
          return HttpResponse.json([original]);
        }),
        http.post("*/api/integrations/github/callback", () => {
          callbackCompleted = true;
          return HttpResponse.json(
            oauthCredential("cred-new", [REQUIRED_SCOPE]),
          );
        }),
        http.put(
          "*/api/chat/sessions/*/credential-selection",
          async ({ request }) => {
            const body = (await request.json()) as {
              selections: Record<string, string>;
            };
            selections.push(body.selections);
            return HttpResponse.json({ status: "ok" });
          },
        ),
      );
      const { onSend } = renderChain();
      const user = userEvent.setup();
      await user.click(
        await screen.findByRole("button", { name: "Change account" }),
      );
      await user.click(await screen.findByRole("button", { name: "Add new" }));
      await user.click(
        await screen.findByRole("button", { name: "Continue with GitHub" }),
      );

      await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
      expect(selections).toEqual([{ github: "cred-new" }]);
      expect(upgradedCredentialID).toBeNull();
    },
  );

  it("surfaces the backend's refusal when the sign-in is a different user", async () => {
    savedCredentials = [
      oauthCredential("cred-a", ["notifications"]),
      oauthCredential("cred-b", ["read:user"]),
    ];
    server.use(
      http.post("*/api/integrations/github/callback", () =>
        HttpResponse.json(
          { detail: "Username mismatch: authenticated as a different user" },
          { status: 400 },
        ),
      ),
    );

    renderChain();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Connect" }));
    const accounts = await screen.findAllByRole("radio");
    await user.click(accounts[1]);
    await user.click(
      screen.getByRole("button", { name: "Update this account" }),
    );
    await user.click(
      await screen.findByRole("button", { name: "Continue with GitHub" }),
    );

    // The sign-in was aimed at cred-b, and a different user behind it is
    // refused rather than stored as yet another account.
    await waitFor(() => expect(upgradedCredentialID).toBe("cred-b"));
    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          description: "Username mismatch: authenticated as a different user",
          variant: "destructive",
        }),
      ),
    );
  });

  it("does not offer a managed account, which the backend refuses to upgrade", async () => {
    savedCredentials = [
      {
        ...oauthCredential("cred-managed", ["notifications"]),
        is_managed: true,
      },
    ];

    renderChain();
    await completeConnectFlow();

    // _prepare_scope_upgrade 400s a managed credential, so offering one makes
    // every Connect click fail and the row permanently unconnectable.
    await waitFor(() => expect(requestedScopes).toBe(REQUIRED_SCOPE));
    expect(upgradedCredentialID).toBeNull();
  });

  it("does not offer an API-key account for an OAuth upgrade", async () => {
    savedCredentials = [
      { id: "cred-key", provider: "github", type: "api_key", title: "GitHub" },
    ];

    renderChain();
    await completeConnectFlow();

    await waitFor(() => expect(requestedScopes).toBe(REQUIRED_SCOPE));
    expect(upgradedCredentialID).toBeNull();
  });
});

describe("copilot Connect card, API-key provider", () => {
  beforeEach(() => {
    requestedScopes = null;
    upgradedCredentialID = null;
    savedCredentials = [];
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([{ name: "openai", description: "Models" }]),
      ),
      http.get("*/api/integrations/providers/system", () =>
        HttpResponse.json([]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(savedCredentials),
      ),
      http.post("*/api/integrations/openai/credentials", async () => {
        const stored = {
          id: "cred-key",
          provider: "openai",
          type: "api_key",
          title: "My key",
        };
        savedCredentials = [...savedCredentials, stored];
        return HttpResponse.json(stored);
      }),
    );
  });

  afterEach(resetStores);

  it("continues the chat after an API key is saved", async () => {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[apiKeyRequirementsPart()]} isStreaming={false} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Connect" }));
    await user.click(await screen.findByText("API Key"));
    await user.type(await screen.findByLabelText("Name"), "My key");
    await user.type(
      await screen.findByPlaceholderText("sk-..."),
      "sk-test-value",
    );
    await user.click(await screen.findByRole("button", { name: "Continue" }));

    // The OAuth branch was fixed first; this is the same stall on the other
    // button of the same dialog.
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
  });
});

describe("copilot Connect card, already satisfied", () => {
  beforeEach(() => {
    requestedScopes = null;
    upgradedCredentialID = null;
    // The state a chat is in after the user connected and later reopened it:
    // every card in the history re-mounts with its credential already there.
    savedCredentials = [
      {
        id: "cred-ok",
        provider: "github",
        type: "oauth2",
        title: "GitHub",
        scopes: [REQUIRED_SCOPE],
      },
      {
        id: "cred-slack",
        provider: "slack",
        type: "oauth2",
        title: "Slack",
        scopes: ["chat:write"],
      },
    ];
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          { name: "github", description: "Issues, pull requests" },
          { name: "slack", description: "Messages" },
        ]),
      ),
      http.get("*/api/integrations/providers/system", () =>
        HttpResponse.json([]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(savedCredentials),
      ),
    );
  });

  afterEach(resetStores);

  it("stays silent when a finished card re-mounts from chat history", async () => {
    const { onSend } = renderChain();

    await settle({ connectedRows: 1 });

    expect(onSend).not.toHaveBeenCalled();
  });

  it("stays silent for two finished cards asking for different services", async () => {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[setupRequirementsPart(), slackRequirementsPart()]}
            isStreaming={false}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await settle({ connectedRows: 2 });

    // The auto-send claim is keyed by provider set, so two cards are two
    // claims — each would fire its own message on mount.
    expect(onSend).not.toHaveBeenCalled();
  });

  // #17253: a reloaded chat mounts every finished card again, connected. In
  // an earlier answer its step has already run, so a Proceed there, or a
  // send of any kind, asks Otto to run that step a second time.
  describe("after a reload", () => {
    beforeEach(trackPickSaves);
    afterEach(stopTrackingPickSaves);

    it("offers no Proceed on a card in an earlier answer", async () => {
      const { onSend } = renderTranscript(chatAfterTheStepRan());

      await settle({ connectedRows: 1 });

      expect(screen.queryByRole("button", { name: "Proceed" })).toBeNull();
      expect(onSend).not.toHaveBeenCalled();
    });

    it("saves nothing for a card in an earlier answer that only mounted again with its account", async () => {
      const written = recordPickSaves();
      const { onSend } = renderTranscript([
        ...chatAfterTheStepRan(),
        ...aWorkflowRunAfterIt(),
      ]);
      const earlier = await findAnswer(`${SESSION_ID}-seq-2`);
      const newest = await findAnswer(`${SESSION_ID}-seq-6`);
      await within(earlier).findByText("Connected");
      const user = userEvent.setup();

      // The newest card's Proceed saves its account and then sends. A save
      // the earlier card made as it mounted went out before that one.
      await user.click(
        await within(newest).findByRole("button", { name: "Proceed" }),
      );

      await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
      expect(written).toEqual([{ slack: "cred-slack" }]);
      expect(pickSaves.started).toBe(1);
    });

    it("saves only the account picked on a card in an earlier answer, not the others it names", async () => {
      savedCredentials = [...savedCredentials, WORK_ACCOUNT];
      const written = recordPickSaves();
      // One step that needs GitHub and Slack. The card names the only Slack
      // account by itself; for GitHub the user picks one of two.
      const chat = chatAfterTheStepRan();
      chat[1] = {
        ...chat[1],
        parts: [
          githubAndSlackRequirementsPart() as ChatMessage["parts"][number],
        ],
      };
      renderTranscript(chat);
      const earlier = await findAnswer(`${SESSION_ID}-seq-2`);
      await within(earlier).findByText("Connected");

      await pickAccount(earlier, "Choose account", /^Work/);

      // Slack may since have been switched to another account on another
      // card, and this card never asked the user about it.
      await waitFor(() => expect(pickSaves.answered).toBe(1));
      expect(written).toEqual([{ github: "cred-work" }]);
    });

    it("saves an account picked on a card in an earlier answer, and sends nothing for it", async () => {
      savedCredentials = [...savedCredentials, WORK_ACCOUNT];
      const written = recordPickSaves();
      const { onSend } = renderTranscript([
        ...chatAfterTheStepRan(),
        ...aWorkflowRunAfterIt(),
      ]);
      const earlier = await findAnswer(`${SESSION_ID}-seq-2`);
      const newest = await findAnswer(`${SESSION_ID}-seq-6`);
      const user = userEvent.setup();

      // Picking an account counts as connecting one, which is what sends a
      // card's follow-up in the newest answer.
      await pickAccount(earlier, "Choose account", /^Work/);

      // The row names Work now, so the tools must run on Work from here on.
      await waitFor(() => expect(pickSaves.answered).toBe(1));
      expect(written).toEqual([{ github: "cred-work" }]);
      expect(
        within(earlier).queryByRole("button", { name: "Proceed" }),
      ).toBeNull();

      // The newest card's Proceed saves its own account and then sends. That
      // save is answered after the one above, so a send the earlier card
      // made after saving would already have gone out by then.
      await user.click(
        await within(newest).findByRole("button", { name: "Proceed" }),
      );
      await waitFor(() => expect(onSend).toHaveBeenCalled());
      expect(onSend).toHaveBeenCalledOnce();
      expect(onSend.mock.calls[0][0]).toContain(
        "proceed with running the workflow",
      );
      expect(written).toEqual([
        { github: "cred-work" },
        { slack: "cred-slack" },
      ]);
    });

    it("sends nothing when the chat moves on while the newest card saves its picks", async () => {
      const release = Promise.withResolvers<void>();
      server.use(
        http.put("*/api/chat/sessions/*/credential-selection", async () => {
          await release.promise;
          return HttpResponse.json({ status: "ok" });
        }),
      );
      const chat = [...chatAfterTheStepRan(), ...aWorkflowRunAfterIt()];
      const { onSend, showMessages } = renderTranscript(chat.slice(0, 2));
      await settle({ connectedRows: 1 });
      const user = userEvent.setup();

      try {
        await user.click(
          await screen.findByRole("button", { name: "Proceed" }),
        );
        await waitFor(() => expect(pickSaves.started).toBe(1));
        // A newer answer lands before the picks are saved, so the card's own
        // answer is now an earlier one.
        showMessages(chat);
      } finally {
        release.resolve();
      }
      await waitFor(() => expect(pickSaves.answered).toBe(1));

      // The newest card's send goes out after that save is answered, so a
      // send the earlier card made once it was saved would already be here.
      const newest = await findAnswer(`${SESSION_ID}-seq-6`);
      await user.click(
        await within(newest).findByRole("button", { name: "Proceed" }),
      );
      await waitFor(() => expect(onSend).toHaveBeenCalled());
      expect(onSend).toHaveBeenCalledOnce();
      expect(onSend.mock.calls[0][0]).toContain(
        "proceed with running the workflow",
      );
    });

    it("saves an account picked on the newest card but never sent, once its answer is an earlier one", async () => {
      savedCredentials = [...savedCredentials, WORK_ACCOUNT];
      const written = recordPickSaves();
      // A trigger card waits for its Proceed, so a pick there sends nothing
      // by itself.
      const chat = chatAfterTheStepRan();
      chat[1] = {
        ...chat[1],
        parts: [triggerPart() as ChatMessage["parts"][number]],
      };
      const { showMessages } = renderTranscript(chat.slice(0, 2));
      const card = await findAnswer(`${SESSION_ID}-seq-2`);

      await pickAccount(card, "Choose account", /^Work/);
      await waitFor(() =>
        expect(
          within(card)
            .getByRole("button", { name: "Proceed" })
            .hasAttribute("disabled"),
        ).toBe(false),
      );
      // The user moves on without Proceed, and the row still names Work.
      showMessages(chat);

      await waitFor(() => expect(pickSaves.answered).toBe(1));
      expect(written).toEqual([{ github: "cred-work" }]);
    });

    it("saves two quick picks on a card in an earlier answer in the order they were made", async () => {
      savedCredentials = [...savedCredentials, WORK_ACCOUNT];
      // The backend keeps whichever pick it writes last. The first pick's
      // save is slow, so a second save sent beside it is written first.
      const firstSaveHeld = Promise.withResolvers<void>();
      const written = recordPickSaves({
        hold: { account: "cred-work", until: firstSaveHeld.promise },
      });
      renderTranscript(chatAfterTheStepRan());
      const earlier = await findAnswer(`${SESSION_ID}-seq-2`);
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      try {
        await pickAccount(earlier, "Choose account", /^Work/);
        await waitFor(() => expect(pickSaves.started).toBe(1));
        await pickAccount(earlier, "Change account", /^GitHub/);
        // Every save the page has sent, except the held one, is written.
        await waitFor(() =>
          expect(pickSaves.answered).toBe(pickSavesSent(fetchSpy) - 1),
        );
      } finally {
        fetchSpy.mockRestore();
        firstSaveHeld.resolve();
      }

      await waitFor(() => expect(pickSaves.answered).toBe(2));
      expect(written).toEqual([{ github: "cred-work" }, { github: "cred-ok" }]);
    });

    it("keeps saving picks on a card in an earlier answer after a save fails", async () => {
      savedCredentials = [...savedCredentials, WORK_ACCOUNT];
      const written = recordPickSaves({ failing: "cred-work" });
      renderTranscript(chatAfterTheStepRan());
      const earlier = await findAnswer(`${SESSION_ID}-seq-2`);

      await pickAccount(earlier, "Choose account", /^Work/);
      await waitFor(() => expect(pickSaves.answered).toBe(1));
      await pickAccount(earlier, "Change account", /^GitHub/);

      await waitFor(() => expect(pickSaves.answered).toBe(2));
      expect(written).toEqual([{ github: "cred-ok" }]);
    });

    it("still offers Proceed on the same card in the newest answer", async () => {
      const { onSend } = renderTranscript(chatAfterTheStepRan().slice(0, 2));
      await settle({ connectedRows: 1 });
      const user = userEvent.setup();

      // The card is still the latest thing Otto asked for, and it has no
      // other way to continue.
      await user.click(await screen.findByRole("button", { name: "Proceed" }));

      await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
      expect(onSend.mock.calls[0][0]).toContain(
        "I've configured the required credentials.",
      );
    });
  });

  // The newest answer is still the newest while Otto writes it, and a
  // follow-up the user sends meanwhile only splits it on screen: the work
  // above the follow-up, the follow-up, the work below.
  describe("in the newest answer, mid-turn", () => {
    it("still offers Proceed while the answer streams", async () => {
      const { onSend } = renderTranscript(chatAfterTheStepRan().slice(0, 2), {
        status: "streaming",
      });
      const user = userEvent.setup();

      // No tool call is pending, so the chain may send mid-turn.
      await user.click(await screen.findByRole("button", { name: "Proceed" }));

      await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
    });

    it("still sends from a card above a follow-up the user sent mid-turn", async () => {
      const { onSend } = renderTranscript(
        answerSplitByAFollowUp(setupRequirementsPart()),
        { status: "streaming" },
      );
      const above = await findAnswer(`${SESSION_ID}-seq-2#seg0`);
      const user = userEvent.setup();

      await user.click(
        await within(above).findByRole("button", { name: "Proceed" }),
      );

      await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
      expect(onSend.mock.calls[0][0]).toContain(
        "I've configured the required credentials.",
      );
    });

    it("sends the answers to a question asked above a follow-up the user sent mid-turn", async () => {
      const { onSend } = renderTranscript(
        answerSplitByAFollowUp(
          questionPart([{ question: "Which region?", keyword: "region" }]),
        ),
        { status: "streaming" },
      );
      const above = await findAnswer(`${SESSION_ID}-seq-2#seg0`);
      const user = userEvent.setup();

      await user.type(await within(above).findByRole("textbox"), "Europe");
      await user.click(
        within(above).getByRole("button", { name: "Send answers" }),
      );

      await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
      expect(onSend.mock.calls[0][0]).toContain("> Which region?\n\nEurope");
      await waitFor(() =>
        expect(screen.queryByText("Answer a few questions")).toBeNull(),
      );
    });
  });
});

describe("copilot Connect card, one tool needing several providers", () => {
  beforeEach(mockTwoProviderBackend);

  afterEach(resetStores);

  it("sends once, after the second provider is connected", async () => {
    const { onSend } = renderTwoProviderChain();

    await connectRemainingProvider();
    // The first account alone must not start the turn: the other row is still
    // visibly unconnected, and the tool needs both.
    await waitFor(() => expect(savedCredentials).toHaveLength(1));
    expect(onSend).not.toHaveBeenCalled();

    await connectRemainingProvider();

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toContain(
      "I've configured the required credentials.",
    );
  });

  it("sends once whichever provider the user connects first", async () => {
    const { onSend } = renderTwoProviderChain();

    // Same assertion from the other end: the order is the user's, the single
    // turn is not.
    const buttons = screen.getAllByRole("button", { name: "Connect" });
    await connectRemainingProvider(buttons.length - 1);
    expect(onSend).not.toHaveBeenCalled();

    await connectRemainingProvider();

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
  });

  it("does not send when the user starts typing an input", async () => {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[partWithInputs()]} isStreaming={false} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await connectRemainingProvider();
    const user = userEvent.setup();
    await user.type(await screen.findByRole("textbox"), "h");

    // An input counts as complete on any non-empty value, so readiness alone
    // would fire the turn mid-word with `{"url":"h"}`.
    await waitFor(() =>
      expect((screen.getByRole("textbox") as HTMLInputElement).value).toBe("h"),
    );
    expect(onSend).not.toHaveBeenCalled();
    await screen.findByRole("button", { name: "Proceed" });
  });

  it("does not send when the user starts typing a clarifying answer", async () => {
    const pending: PendingQuestions = {
      dockId: "m1:call-ask_question",
      callIds: ["call-ask_question"],
      questions: [{ question: "Which region?", keyword: "region" }],
    };
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <PendingQuestionsContext.Provider value={pending}>
            <ToolChain
              parts={[setupRequirementsPart(), questionPart(pending.questions)]}
              isStreaming={false}
            />
          </PendingQuestionsContext.Provider>
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await completeConnectFlow();
    const user = userEvent.setup();
    await user.type(await screen.findByRole("textbox"), "W");

    // Answers are complete on the first character too, and the dock's own
    // "add answers" review step would never run.
    await waitFor(() =>
      expect((screen.getByRole("textbox") as HTMLInputElement).value).toBe("W"),
    );
    expect(onSend).not.toHaveBeenCalled();
  });

  it("offers one Proceed when a card also collects inputs", async () => {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[partWithInputs(), slackRequirementsPart()]}
            isStreaming={false}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await connectRemainingProvider();
    await connectRemainingProvider();

    // Connecting every account does not finish a chain that also owes inputs;
    // the user completes it through the chain's single Proceed.
    await screen.findByRole("button", { name: "Proceed" });
    expect(onSend).not.toHaveBeenCalled();
  });

  it("stays silent and names what is left when the user connects one and stops", async () => {
    const { onSend } = renderTwoProviderChain();

    await connectRemainingProvider();

    await screen.findByText(/Still to connect:/);
    await new Promise((resolve) => setTimeout(resolve, 300));
    expect(onSend).not.toHaveBeenCalled();
  });
});

describe("copilot Connect card, an account and an MCP server in one step", () => {
  beforeEach(mockAccountAndMcpBackend);

  afterEach(resetStores);

  it("sends one message when the MCP server is connected last", async () => {
    const { onSend } = renderAccountAndMcpChain();

    await connectRemainingProvider();
    await waitFor(() => expect(savedCredentials).toHaveLength(1));
    expect(onSend).not.toHaveBeenCalled();

    await connectMcpServer();

    // The MCP card used to send its own "retry" message here, and the chain
    // then sent the account's confirmation as a second one, which Otto
    // received queued behind the first.
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toContain(
      "I've configured the required credentials.",
    );
    expect(onSend.mock.calls[0][0]).toContain(MCP_RETRY);
    await screen.findByText(/Connected\. Continuing/);
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("sends one message when the MCP server is connected first", async () => {
    const { onSend } = renderAccountAndMcpChain();

    await connectMcpServer();
    await waitFor(() => expect(savedCredentials).toHaveLength(1));
    await screen.findByText("Connected");
    // The account row is still unconnected, and the step needs both.
    expect(onSend).not.toHaveBeenCalled();

    await connectRemainingProvider();

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toContain(
      "I've configured the required credentials.",
    );
    expect(onSend.mock.calls[0][0]).toContain(MCP_RETRY);
    await screen.findByText(/Connected\. Continuing/);
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("leaves an already connected MCP server out of the account's message", async () => {
    savedCredentials = [
      {
        id: "cred-mcp",
        provider: "mcp",
        type: "oauth2",
        title: "Notion",
        host: MCP_SERVER,
        scopes: [],
      },
    ];
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[setupRequirementsPart(), unconnectedMcpPart()]}
            isStreaming={false}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await connectRemainingProvider();

    // The server was connected before this chat, so it owes Otto no line.
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toContain(
      "I've configured the required credentials.",
    );
    expect(onSend.mock.calls[0][0]).not.toContain(MCP_RETRY);
  });

  it("gives an MCP card that lands after the send a Proceed", async () => {
    const onSend = vi.fn();
    const { rerender } = render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[setupRequirementsPart()]} isStreaming={true} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );
    await connectRemainingProvider();
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));

    rerender(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[setupRequirementsPart(), unconnectedMcpPart()]}
            isStreaming={true}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );
    await connectMcpServer();

    // The chain's single send is spent and the card no longer sends on its
    // own, so the user needs a button to tell Otto the server is connected.
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Proceed" }));
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(2));
    expect(onSend.mock.calls[1][0]).toContain(MCP_RETRY);
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Proceed" })).toBeNull(),
    );
    expect(onSend).toHaveBeenCalledTimes(2);
  });

  it("does not repeat the MCP server's line after a Proceed sends it", async () => {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[triggerPart(), unconnectedMcpPart()]}
            isStreaming={false}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await connectRemainingProvider();
    await connectMcpServer();
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Proceed" }));
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toContain(MCP_RETRY);

    // The trigger card leaves the chain once sent, which leaves the MCP row
    // as the chain's only card. Still counting as just connected, it would
    // auto-send its line a second time, from the effect that runs once the
    // row is alone; draining that effect's send before asserting.
    await waitFor(() =>
      expect(screen.getAllByText("Connected")).toHaveLength(1),
    );
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("still sends the MCP server's message when it is the only sign-in", async () => {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[unconnectedMcpPart()]} isStreaming={false} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await connectMcpServer();

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toBe(MCP_RETRY);
    // The line is sent, so a Proceed left here would only send it again. A
    // regressed Proceed appears before the send and stays, so waiting for it
    // to be gone cannot pass early.
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Proceed" })).toBeNull(),
    );
    expect(onSend).toHaveBeenCalledTimes(1);
  });
});

describe("copilot Connect card, a card that arrives after the first connection", () => {
  beforeEach(mockTwoProviderBackend);

  afterEach(resetStores);

  it("waits for a tool that is still running, then sends once", async () => {
    const onSend = vi.fn();
    const { rerender } = render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[setupRequirementsPart(), pendingSlackCallPart()]}
            isStreaming={true}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );

    await connectRemainingProvider();

    // The running tool has not said what it needs yet, so sending here spends
    // the chain's one turn before that card exists.
    await screen.findByText("Connected");
    expect(onSend).not.toHaveBeenCalled();

    rerender(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[setupRequirementsPart(), slackRequirementsPart()]}
            isStreaming={false}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );
    await connectRemainingProvider();

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toContain(
      "I've configured the required credentials.",
    );
  });

  it("gives a card that lands after the send a Proceed to continue from", async () => {
    const onSend = vi.fn();
    const { rerender } = render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain parts={[setupRequirementsPart()]} isStreaming={true} />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );
    await connectRemainingProvider();
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));

    rerender(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[setupRequirementsPart(), slackRequirementsPart()]}
            isStreaming={true}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );
    await connectRemainingProvider();

    // The chain's single send is spent, so without a button this card is
    // connected with nothing to press.
    await screen.findByRole("button", { name: "Proceed" });
    expect(onSend).toHaveBeenCalledTimes(1);
  });
});

describe("copilot Connect card, a saved credential the provider refused", () => {
  // The credential the provider refused. It stays on file and still has every
  // scope the card asks for, which is what made the row read Connected.
  const refused = oauthCredential("cred-refused", [REQUIRED_SCOPE]);
  let picked: Record<string, string> | null = null;

  beforeEach(() => {
    requestedScopes = null;
    upgradedCredentialID = null;
    picked = null;
    savedCredentials = [refused];
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([{ name: "github", description: "Repositories" }]),
      ),
      http.get("*/api/integrations/providers/system", () =>
        HttpResponse.json([]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(savedCredentials),
      ),
      http.get("*/api/integrations/github/login", ({ request }) => {
        const query = new URL(request.url).searchParams;
        requestedScopes = query.get("scopes") ?? "";
        upgradedCredentialID = query.get("credential_id");
        return HttpResponse.json({
          login_url: "https://github.com/login/oauth/authorize",
          state_token: "state-token",
        });
      }),
      // A re-auth aimed at an account updates it in place: same id, new token.
      http.post("*/api/integrations/github/callback", () =>
        HttpResponse.json(refused),
      ),
      http.put(
        "*/api/chat/sessions/:sessionId/credential-selection",
        async ({ request }) => {
          picked = ((await request.json()) as { selections: typeof picked })
            .selections;
          return HttpResponse.json({});
        },
      ),
    );
  });

  afterEach(resetStores);

  function renderRefusedChain() {
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[refusedRequirementsPart("cred-refused")]}
            isStreaming={false}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );
    return { onSend };
  }

  async function reconnect() {
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Reconnect" }));
    await user.click(
      await screen.findByRole("button", { name: "Continue with GitHub" }),
    );
  }

  it("offers Reconnect instead of calling the refused credential Connected", async () => {
    const { onSend } = renderRefusedChain();

    // A Connected row renders no button, so it would leave nothing to click.
    await screen.findByRole("button", { name: "Reconnect" });
    expect(screen.queryByText("Connected")).toBeNull();
    expect(onSend).not.toHaveBeenCalled();
  });

  it("renews the refused account in place and continues the chat", async () => {
    const { onSend } = renderRefusedChain();

    await reconnect();

    await waitFor(() => expect(upgradedCredentialID).toBe("cred-refused"));
    // The renewal keeps the credential's id, so a card that kept treating that
    // id as refused would never send.
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(picked).toEqual({ github: "cred-refused" });
    await screen.findByText("Connected. Continuing…");
  });

  it("runs on the new credential, not the refused one, when the sign-in adds one", async () => {
    server.use(
      http.post("*/api/integrations/github/callback", () => {
        const added = oauthCredential("cred-new", [REQUIRED_SCOPE]);
        savedCredentials = [refused, added];
        return HttpResponse.json(added);
      }),
    );
    const { onSend } = renderRefusedChain();

    await reconnect();

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(picked).toEqual({ github: "cred-new" });
  });

  it("keeps asking when the sign-in comes back without the access needed", async () => {
    // A narrower grant is stored beside the refused account, not over it.
    server.use(
      http.post("*/api/integrations/github/callback", () => {
        const short = oauthCredential("cred-short", []);
        savedCredentials = [refused, short];
        return HttpResponse.json(short);
      }),
    );
    const { onSend } = renderRefusedChain();

    await reconnect();

    await waitFor(() => expect(savedCredentials).toHaveLength(2));
    await screen.findByRole("button", { name: "Reconnect" });
    expect(onSend).not.toHaveBeenCalled();
  });

  it("finds the new account when the sign-in does not report it", async () => {
    // Any 2xx other than 200 reaches the dialog as a success with no
    // credential, which is how a credential-less approval looks to the row.
    server.use(
      http.post("*/api/integrations/github/callback", () => {
        const added = oauthCredential("cred-new", [REQUIRED_SCOPE]);
        savedCredentials = [refused, added];
        return HttpResponse.json(added, { status: 201 });
      }),
    );
    const { onSend } = renderRefusedChain();

    await reconnect();

    // Nothing says the refused account was renewed, so it must not be the
    // one the chat runs on.
    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(picked).toEqual({ github: "cred-new" });
  });

  describe("when another saved account still works", () => {
    const other = {
      ...oauthCredential("cred-other", [REQUIRED_SCOPE]),
      title: "Work",
    };

    beforeEach(() => {
      savedCredentials = [refused, other];
    });

    it("asks instead of quietly running on the other account", async () => {
      const { onSend } = renderRefusedChain();
      const user = userEvent.setup();

      // A different account can mean posting as someone else, so the row
      // waits for the user rather than calling the other one Connected.
      await user.click(
        await screen.findByRole("button", { name: "Reconnect" }),
      );
      expect(screen.queryByText("Connected")).toBeNull();
      // Only the account that still works is offered; the refused one is
      // renewed through Add new.
      expect(screen.queryByRole("radio", { name: /^GitHub/ })).toBeNull();
      await user.click(await screen.findByRole("radio", { name: /^Work/ }));
      await user.click(
        screen.getByRole("button", { name: "Use this account" }),
      );

      await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
      expect(picked).toEqual({ github: "cred-other" });
    });

    it("runs on the renewed account once the user signs in to it again", async () => {
      const { onSend } = renderRefusedChain();
      const user = userEvent.setup();

      await user.click(
        await screen.findByRole("button", { name: "Reconnect" }),
      );
      await user.click(await screen.findByRole("button", { name: "Add new" }));
      await user.click(
        await screen.findByRole("button", { name: "Continue with GitHub" }),
      );

      // Both accounts now fit; the one just signed in to is the one meant.
      await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
      expect(picked).toEqual({ github: "cred-refused" });
    });

    it("waits for a newly added account rather than falling back to the other", async () => {
      // The first list fetched after the sign-in predates it, as when the
      // provider list lags the callback.
      let staleFetches = 0;
      server.use(
        http.post("*/api/integrations/github/callback", () => {
          const added = oauthCredential("cred-new", [REQUIRED_SCOPE]);
          savedCredentials = [refused, other, added];
          staleFetches = 1;
          return HttpResponse.json(added);
        }),
        http.get("*/api/integrations/credentials", () => {
          if (staleFetches === 0) return HttpResponse.json(savedCredentials);
          staleFetches -= 1;
          return HttpResponse.json([refused, other]);
        }),
      );
      const { onSend } = renderRefusedChain();
      const user = userEvent.setup();

      await user.click(
        await screen.findByRole("button", { name: "Reconnect" }),
      );
      await user.click(await screen.findByRole("button", { name: "Add new" }));
      await user.click(
        await screen.findByRole("button", { name: "Continue with GitHub" }),
      );

      // Until the list shows the new account, the other one is the only
      // usable match; picking it then would run on an account nobody chose.
      await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
      expect(picked).toEqual({ github: "cred-new" });
    });
  });
});

function refusedRequirementsPart(credentialID: string): MessagePart {
  const part = setupRequirementsPart() as unknown as {
    output: Record<string, unknown>;
  };
  return {
    ...part,
    output: {
      ...part.output,
      message: "The saved Github credential 'GitHub' could not be refreshed.",
      rejection: {
        provider: "github",
        detail: "invalid_grant",
        status_code: null,
        credential_id: credentialID,
        credential_title: "GitHub",
      },
    },
  } as unknown as MessagePart;
}

describe("named HTTP connections", () => {
  beforeEach(() => {
    savedCredentials = [];
    server.use(
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          { name: "http", description: "Generic HTTP requests" },
        ]),
      ),
      http.get("*/api/integrations/providers/system", () =>
        HttpResponse.json([]),
      ),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(savedCredentials),
      ),
    );
  });
  afterEach(resetStores);

  it("saves Linear and GitHub keys for their own hosts and continues only after both are ready", async () => {
    const created: Record<string, unknown>[] = [];
    const selections: Record<string, string>[] = [];
    server.use(
      http.put(
        "*/api/chat/sessions/*/credential-selection",
        async ({ request }) => {
          const body = (await request.json()) as {
            selections: Record<string, string>;
          };
          selections.push(body.selections);
          return HttpResponse.json({ status: "ok" });
        },
      ),
      http.post("*/api/integrations/http/credentials", async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        created.push(body);
        const stored = {
          id: `http-${created.length}`,
          provider: "http",
          type: "host_scoped",
          title: body.title,
          host: body.host,
        };
        savedCredentials.push(stored);
        return HttpResponse.json(stored);
      }),
    );
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[
              httpRequirementsPart("linear", "https://api.linear.app/graphql"),
              httpRequirementsPart("github", "https://api.github.com/repos"),
            ]}
            isStreaming={false}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );
    const user = userEvent.setup();
    await screen.findByText("0 of 2 ready");
    await screen.findByText("Linear");
    await screen.findByText("GitHub");
    const buttons = await screen.findAllByRole("button", {
      name: "Connect",
    });
    await user.click(buttons[0]);
    await screen.findByRole("heading", { name: "Connect Linear" });
    expect(screen.queryByLabelText("Host")).toBeNull();
    await user.type(screen.getByLabelText("API key"), "linear-test-key");
    await user.click(screen.getByRole("button", { name: "Save and connect" }));
    await screen.findByText("1 of 2 ready");
    expect(onSend).not.toHaveBeenCalled();
    expect(created[0]).toMatchObject({
      provider: "http",
      type: "host_scoped",
      host: "api.linear.app",
      headers: { Authorization: "linear-test-key" },
    });
    await user.click(await screen.findByRole("button", { name: "Connect" }));
    await screen.findByRole("heading", { name: "Connect GitHub" });
    await user.type(screen.getByLabelText("API key"), "github-test-key");
    await user.click(screen.getByRole("button", { name: "Save and connect" }));
    await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
    expect(created[1]).toMatchObject({
      host: "api.github.com",
      headers: { Authorization: "Bearer github-test-key" },
    });
    expect(selections).toEqual([
      { "http:api.linear.app": "http-1" },
      { "http:api.github.com": "http-2" },
    ]);
  });

  it.each(["api.linear.app", "api.github.com"])(
    "validates a returned host-scoped credential when the list stays stale (host: %s)",
    async (host) => {
      const selections: Record<string, string>[] = [];
      server.use(
        http.get("*/api/integrations/credentials", () => HttpResponse.json([])),
        http.post("*/api/integrations/http/credentials", () =>
          HttpResponse.json({
            id: "new-key",
            provider: "http",
            type: "host_scoped",
            title: "Linear",
            host,
          }),
        ),
        http.put(
          "*/api/chat/sessions/*/credential-selection",
          async ({ request }) => {
            const body = (await request.json()) as {
              selections: Record<string, string>;
            };
            selections.push(body.selections);
            return HttpResponse.json({ status: "ok" });
          },
        ),
      );
      const { onSend } = renderChain([
        httpRequirementsPart("linear", "https://api.linear.app/graphql"),
      ]);
      const user = userEvent.setup();
      await user.click(await screen.findByRole("button", { name: "Connect" }));
      await user.type(
        await screen.findByLabelText("API key"),
        "linear-test-key",
      );
      await user.click(
        screen.getByRole("button", { name: "Save and connect" }),
      );

      if (host === "api.linear.app") {
        await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
        expect(selections).toEqual([{ "http:api.linear.app": "new-key" }]);
      } else {
        expect(
          await screen.findByText(
            "That account is missing the access this needs. Try connecting again.",
          ),
        ).toBeDefined();
        expect(onSend).not.toHaveBeenCalled();
        expect(selections).toEqual([]);
      }
    },
  );

  it("keeps the key form open after a failed save and lets the user retry", async () => {
    let attempts = 0;
    server.use(
      http.post("*/api/integrations/http/credentials", async ({ request }) => {
        attempts += 1;
        if (attempts === 1)
          return HttpResponse.json(
            { detail: "Could not save connection" },
            { status: 400 },
          );
        const body = (await request.json()) as Record<string, unknown>;
        const stored = {
          id: "retry-key",
          provider: "http",
          type: "host_scoped",
          host: body.host,
          title: body.title,
        };
        savedCredentials.push(stored);
        return HttpResponse.json(stored);
      }),
    );
    const onSend = vi.fn();
    render(
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ToolChain
            parts={[
              httpRequirementsPart("linear", "https://api.linear.app/graphql"),
            ]}
            isStreaming={false}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>,
    );
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Connect" }));
    await user.type(await screen.findByLabelText("API key"), "retry-test-key");
    await user.click(screen.getByRole("button", { name: "Save and connect" }));
    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Couldn't connect Linear" }),
      ),
    );
    expect(onSend).not.toHaveBeenCalled();
    expect((screen.getByLabelText("API key") as HTMLInputElement).value).toBe(
      "retry-test-key",
    );
    await user.click(
      await screen.findByRole("button", { name: "Save and connect" }),
    );
    await waitFor(() => expect(onSend).toHaveBeenCalledOnce());
    expect(attempts).toBe(2);
  });
});

function httpRequirementsPart(name: string, url: string): MessagePart {
  return {
    type: "tool-run_block",
    toolCallId: `http-${name}`,
    state: "output-available",
    input: { block_name: "HttpSendRequestBlock" },
    output: {
      type: "setup_requirements",
      message: "Connect your apps to continue.",
      session_id: SESSION_ID,
      setup_info: {
        agent_id: `block-${name}`,
        agent_name: "HTTP request",
        requirements: {},
        user_readiness: {
          has_all_credentials: false,
          missing_credentials: {
            credentials: {
              provider: "http",
              types: ["host_scoped"],
              discriminator: "url",
              discriminator_values: [url],
            },
          },
        },
      },
    },
  } as unknown as MessagePart;
}

function renderChain(parts: MessagePart[] = [setupRequirementsPart()]) {
  const onSend = vi.fn();
  const utils = render(
    <CredentialsProvider>
      <CopilotChatActionsProvider onSend={onSend}>
        <ToolChain parts={parts} isStreaming={false} />
      </CopilotChatActionsProvider>
    </CredentialsProvider>,
  );
  return { onSend, ...utils };
}

// Saves of the chat's credential picks, counted as they start and as they are
// answered. A card saves its picks just before it sends its follow-up, or on
// their own in an earlier answer, which sends nothing.
const pickSaves = { started: 0, answered: 0 };

function isPickSave(request: Request) {
  return (
    request.method === "PUT" &&
    new URL(request.url).pathname.endsWith("/credential-selection")
  );
}

function onRequestStart({ request }: { request: Request }) {
  if (isPickSave(request)) pickSaves.started += 1;
}

function onRequestEnd({ request }: { request: Request }) {
  if (isPickSave(request)) pickSaves.answered += 1;
}

function trackPickSaves() {
  pickSaves.started = 0;
  pickSaves.answered = 0;
  server.events.on("request:start", onRequestStart);
  server.events.on("request:end", onRequestEnd);
}

function stopTrackingPickSaves() {
  server.events.removeListener("request:start", onRequestStart);
  server.events.removeListener("request:end", onRequestEnd);
}

/** A second GitHub account that also fits the card, so the card names
 *  neither until one is picked. */
const WORK_ACCOUNT = {
  ...oauthCredential("cred-work", [REQUIRED_SCOPE]),
  title: "Work",
};

/** The picks the backend writes, `{provider: credential id}` each, in the
 *  order it writes them. A save for `hold.account` waits for `hold.until`;
 *  one for `failing` is answered with an error and writes nothing. */
function recordPickSaves({
  hold,
  failing,
}: {
  hold?: { account: string; until: Promise<void> };
  failing?: string;
} = {}) {
  const written: Record<string, string>[] = [];
  server.use(
    http.put(
      "*/api/chat/sessions/*/credential-selection",
      async ({ request }) => {
        const { selections } = (await request.json()) as {
          selections: Record<string, string>;
        };
        const accounts = Object.values(selections);
        if (failing && accounts.includes(failing)) {
          return HttpResponse.json({ detail: "Save failed" }, { status: 500 });
        }
        if (hold && accounts.includes(hold.account)) await hold.until;
        written.push(selections);
        return HttpResponse.json({ status: "ok" });
      },
    ),
  );
  return written;
}

/** Picks `account` on the card in `answer`, opening its account picker with
 *  the button named `opener`. */
async function pickAccount(
  answer: HTMLElement,
  opener: string,
  account: RegExp,
) {
  const user = userEvent.setup();
  await user.click(await within(answer).findByRole("button", { name: opener }));
  await user.click(await screen.findByRole("radio", { name: account }));
  await user.click(screen.getByRole("button", { name: "Use this account" }));
}

/** Saves of the chat's credential picks the page has sent so far, counted as
 *  it calls `fetch`, before the mock server sees them. */
function pickSavesSent(fetchSpy: MockInstance<typeof fetch>) {
  return fetchSpy.mock.calls.filter(
    ([input, init]) =>
      init?.method === "PUT" && String(input).endsWith("/credential-selection"),
  ).length;
}

/** One answer's row in the transcript, or one part of an answer a mid-turn
 *  follow-up split. */
async function findAnswer(messageID: string): Promise<HTMLElement> {
  return waitFor(() => {
    const row = document.querySelector<HTMLElement>(
      `[data-message-id="${messageID}"]`,
    );
    if (!row) throw new Error(`No transcript row for ${messageID}`);
    return row;
  });
}

/** The chat as it loads from history, or as it streams: every row, and the
 *  card wherever its answer falls among them. */
function renderTranscript(
  messages: ChatMessage[],
  { status = "ready" }: { status?: "ready" | "streaming" } = {},
) {
  const onSend = vi.fn();
  function transcript(shown: ChatMessage[]) {
    return (
      <CredentialsProvider>
        <CopilotChatActionsProvider onSend={onSend}>
          <ChatMessagesContainer
            messages={shown}
            status={status}
            error={undefined}
            isLoading={false}
            sessionID={SESSION_ID}
            queuedMessages={[]}
            hasMoreMessages={false}
            isLoadingMore={false}
            onLoadMore={vi.fn()}
            onRetry={vi.fn()}
          />
        </CopilotChatActionsProvider>
      </CredentialsProvider>
    );
  }
  const utils = render(transcript(messages));
  return {
    onSend,
    ...utils,
    /** The same page, once the chat has moved on to `shown`. */
    showMessages: (shown: ChatMessage[]) => utils.rerender(transcript(shown)),
  };
}

/** The ask, the answer with the card, the follow-up the card sent once the
 *  account was connected, and the answer that ran the step. */
function chatAfterTheStepRan(): ChatMessage[] {
  return [
    {
      id: `${SESSION_ID}-seq-1`,
      role: "user",
      parts: [{ type: "text", text: "Read my latest pull request." }],
    },
    {
      id: `${SESSION_ID}-seq-2`,
      role: "assistant",
      parts: [setupRequirementsPart() as ChatMessage["parts"][number]],
    },
    {
      id: `${SESSION_ID}-seq-3`,
      role: "user",
      parts: [
        {
          type: "text",
          text: "I've configured the required credentials. Please re-run this step now.",
        },
      ],
    },
    {
      id: `${SESSION_ID}-seq-4`,
      role: "assistant",
      parts: [{ type: "text", text: "Your latest pull request adds a page." }],
    },
  ];
}

/** The next ask, and the newest answer: a workflow run with a card for
 *  Slack, which the user already has connected. */
function aWorkflowRunAfterIt(): ChatMessage[] {
  return [
    {
      id: `${SESSION_ID}-seq-5`,
      role: "user",
      parts: [{ type: "text", text: "Now post it to Slack." }],
    },
    {
      id: `${SESSION_ID}-seq-6`,
      role: "assistant",
      parts: [
        {
          ...(slackRequirementsPart() as unknown as Record<string, unknown>),
          type: "tool-run_agent",
          toolCallId: "call-run_agent-slack",
          input: { username_agent_slug: "me/post-to-slack" },
        } as unknown as ChatMessage["parts"][number],
      ],
    },
  ];
}

/** The newest answer while Otto still writes it, split on screen by a
 *  follow-up the user sent meanwhile: `above` drew before the follow-up was
 *  taken in, and the reply goes on below it. */
function answerSplitByAFollowUp(above: MessagePart): ChatMessage[] {
  return [
    {
      id: `${SESSION_ID}-seq-1`,
      role: "user",
      parts: [{ type: "text", text: "Read my latest pull request." }],
    },
    {
      id: `${SESSION_ID}-seq-2`,
      role: "assistant",
      parts: [
        above,
        {
          type: "data-pending-drained",
          id: "hint-1",
          data: {
            drainedCount: 1,
            messages: [{ id: "pending-1", content: "Check its CI too." }],
          },
        },
        { type: "text", text: "Checking the CI as well.", state: "streaming" },
      ] as ChatMessage["parts"],
    },
  ];
}

async function completeConnectFlow() {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: "Connect" }));
  await user.click(
    await screen.findByRole("button", { name: "Continue with GitHub" }),
  );
}

/** What the backend returns when a block needs a credential the user's
 *  connected account doesn't cover — here GitHub without the `repo` scope. */
function setupRequirementsPart(types = ["oauth2"]): MessagePart {
  return {
    type: "tool-run_block",
    toolCallId: "call-run_block",
    state: "output-available",
    input: { block_name: "GithubReadPullRequestBlock" },
    output: {
      type: "setup_requirements",
      message: "Connect GitHub to continue.",
      session_id: SESSION_ID,
      setup_info: {
        agent_id: "block-1",
        agent_name: "Github Read Pull Request",
        requirements: {},
        user_readiness: {
          has_all_credentials: false,
          missing_credentials: {
            credentials: {
              provider: "github",
              types,
              scopes: [REQUIRED_SCOPE],
            },
          },
        },
      },
    },
  } as unknown as MessagePart;
}

/** Waits on a positive signal — every row resolved against the refreshed
 *  credential list — so a "nothing was sent" assertion cannot pass merely
 *  because nothing has mounted yet. */
async function settle({ connectedRows }: { connectedRows: number }) {
  await screen.findByText(/Connect your apps|Your apps are connected/);
  await waitFor(() =>
    expect(screen.getAllByText("Connected")).toHaveLength(connectedRows),
  );
}

function slackRequirementsPart(): MessagePart {
  const github = setupRequirementsPart() as unknown as Record<string, unknown>;
  return {
    ...github,
    toolCallId: "call-run_block-slack",
    output: {
      ...(github.output as Record<string, unknown>),
      setup_info: {
        agent_id: "block-2",
        agent_name: "Slack Post Message",
        requirements: {},
        user_readiness: {
          has_all_credentials: false,
          missing_credentials: {
            credentials: {
              provider: "slack",
              types: ["oauth2"],
              scopes: ["chat:write"],
            },
          },
        },
      },
    },
  } as unknown as MessagePart;
}
/** One block that needs both GitHub and Slack. */
function githubAndSlackRequirementsPart(): MessagePart {
  const base = setupRequirementsPart() as unknown as Record<string, unknown>;
  const output = base.output as Record<string, unknown>;
  const setup = output.setup_info as Record<string, unknown>;
  return {
    ...base,
    output: {
      ...output,
      setup_info: {
        ...setup,
        user_readiness: {
          has_all_credentials: false,
          missing_credentials: {
            github_credentials: {
              provider: "github",
              types: ["oauth2"],
              scopes: [REQUIRED_SCOPE],
            },
            slack_credentials: {
              provider: "slack",
              types: ["oauth2"],
              scopes: ["chat:write"],
            },
          },
        },
      },
    },
  } as unknown as MessagePart;
}
function apiKeyRequirementsPart(): MessagePart {
  const github = setupRequirementsPart() as unknown as Record<string, unknown>;
  return {
    ...github,
    toolCallId: "call-run_block-openai",
    output: {
      ...(github.output as Record<string, unknown>),
      setup_info: {
        agent_id: "block-3",
        agent_name: "OpenAI Completion",
        requirements: {},
        user_readiness: {
          has_all_credentials: false,
          missing_credentials: {
            credentials: { provider: "openai", types: ["api_key"] },
          },
        },
      },
    },
  } as unknown as MessagePart;
}
function oauthCredential(id: string, scopes: string[]) {
  return { id, provider: "github", type: "oauth2", title: "GitHub", scopes };
}
function resetStores() {
  cleanup();
  useCopilotUIStore.setState({ initialPrompt: null, sentMessageCount: 0 });
  // The auto-send claim is a module singleton keyed by session id, so a later
  // test reusing SESSION_ID would silently dismiss its own card.
  useConnectedProvidersStore.getState().clearSession(SESSION_ID);
}
function partWithInputs(): MessagePart {
  const base = setupRequirementsPart() as unknown as Record<string, unknown>;
  const output = base.output as Record<string, unknown>;
  const setup = output.setup_info as Record<string, unknown>;
  return {
    ...base,
    output: {
      ...output,
      setup_info: {
        ...setup,
        requirements: {
          inputs: [
            { name: "url", title: "URL", type: "string", required: true },
          ],
        },
      },
    },
  } as unknown as MessagePart;
}

function triggerPart(): MessagePart {
  const base = setupRequirementsPart() as unknown as Record<string, unknown>;
  return {
    ...base,
    type: "tool-setup_agent_webhook_trigger",
    toolCallId: "call-setup_agent_webhook_trigger",
  } as unknown as MessagePart;
}
function renderTwoProviderChain() {
  const onSend = vi.fn();
  const utils = render(
    <CredentialsProvider>
      <CopilotChatActionsProvider onSend={onSend}>
        <ToolChain
          parts={[setupRequirementsPart(), slackRequirementsPart()]}
          isStreaming={false}
        />
      </CopilotChatActionsProvider>
    </CredentialsProvider>,
  );
  return { onSend, ...utils };
}

/** Connects one row by position. The list shrinks as rows resolve, so index 0
 *  is "the next one still unconnected". */
async function connectRemainingProvider(index = 0) {
  const user = userEvent.setup();
  const buttons = await screen.findAllByRole("button", { name: "Connect" });
  await user.click(buttons[index]);
  await user.click(
    await screen.findByRole("button", { name: /^Continue with / }),
  );
}

function questionPart(questions: PendingQuestions["questions"]): MessagePart {
  return {
    type: "tool-ask_question",
    toolCallId: "call-ask_question",
    state: "output-available",
    input: {},
    output: { type: "agent_builder_clarification_needed", questions },
  } as unknown as MessagePart;
}
function mcpPart(): MessagePart {
  return {
    type: "tool-run_mcp_tool",
    toolCallId: "call-run_mcp_tool",
    state: "output-available",
    input: { server_url: "https://mcp.notion.com/mcp" },
    output: {
      type: "setup_requirements",
      message: "Connect Notion to continue.",
      session_id: SESSION_ID,
      setup_info: {
        agent_id: "https://mcp.notion.com/mcp",
        agent_name: "notion.com",
        requirements: {},
        user_readiness: { has_all_credentials: true },
      },
    },
  } as unknown as MessagePart;
}

const MCP_SERVER = "https://mcp.notion.com/mcp";
const MCP_RETRY = "I've connected the integration. Please retry the same call.";

function unconnectedMcpPart(): MessagePart {
  const base = mcpPart() as unknown as Record<string, unknown>;
  const output = base.output as Record<string, unknown>;
  const setup = output.setup_info as Record<string, unknown>;
  return {
    ...base,
    output: {
      ...output,
      setup_info: {
        ...setup,
        user_readiness: { has_all_credentials: false },
      },
    },
  } as unknown as MessagePart;
}

function mockAccountAndMcpBackend() {
  mockTwoProviderBackend();
  server.use(
    http.post("*/api/mcp/oauth/login", () =>
      HttpResponse.json({
        login_url: "https://mcp.notion.com/authorize",
        state_token: "mcp-state",
      }),
    ),
    http.post("*/api/mcp/oauth/callback", () => {
      const stored = {
        id: "cred-mcp",
        provider: "mcp",
        type: "oauth2",
        title: "Notion",
        host: MCP_SERVER,
        scopes: [],
      };
      savedCredentials = [...savedCredentials, stored];
      return HttpResponse.json(stored);
    }),
  );
}

function renderAccountAndMcpChain() {
  const onSend = vi.fn();
  const utils = render(
    <CredentialsProvider>
      <CopilotChatActionsProvider onSend={onSend}>
        <ToolChain
          parts={[setupRequirementsPart(), unconnectedMcpPart()]}
          isStreaming={false}
        />
      </CopilotChatActionsProvider>
    </CredentialsProvider>,
  );
  return { onSend, ...utils };
}

/** The MCP row comes after the account rows, so its Connect is the last one,
 *  and it signs in straight away, without the account dialog. */
async function connectMcpServer() {
  const user = userEvent.setup();
  const buttons = await screen.findAllByRole("button", { name: "Connect" });
  await user.click(buttons[buttons.length - 1]);
}

function widerScopePart(): MessagePart {
  const base = setupRequirementsPart() as unknown as Record<string, unknown>;
  const output = base.output as Record<string, unknown>;
  const setup = output.setup_info as Record<string, unknown>;
  return {
    ...base,
    toolCallId: "call-run_block-wider",
    output: {
      ...output,
      setup_info: {
        ...setup,
        user_readiness: {
          has_all_credentials: false,
          missing_credentials: {
            credentials: {
              provider: "github",
              types: ["oauth2"],
              scopes: [REQUIRED_SCOPE, "workflow"],
            },
          },
        },
      },
    },
  } as unknown as MessagePart;
}

function mockTwoProviderBackend() {
  savedCredentials = [];
  server.use(
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json([
        { name: "github", description: "Repositories" },
        { name: "slack", description: "Messages" },
      ]),
    ),
    http.get("*/api/integrations/providers/system", () =>
      HttpResponse.json([]),
    ),
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json(savedCredentials),
    ),
    http.get("*/api/integrations/:provider/login", () =>
      HttpResponse.json({
        login_url: "https://example.com/oauth/authorize",
        state_token: "state-token",
      }),
    ),
    http.post("*/api/integrations/:provider/callback", ({ params }) => {
      const provider = String(params.provider);
      const stored = {
        id: `cred-${provider}`,
        provider,
        type: "oauth2",
        title: provider,
        scopes: provider === "github" ? [REQUIRED_SCOPE] : ["chat:write"],
      };
      savedCredentials = [...savedCredentials, stored];
      return HttpResponse.json(stored);
    }),
  );
}

/** The Slack tool call before its result arrives, under the id its setup card
 *  later lands on. */
function pendingSlackCallPart(): MessagePart {
  return {
    type: "tool-run_block",
    toolCallId: "call-run_block-slack",
    state: "input-available",
    input: { block_name: "SlackPostMessageBlock" },
  } as unknown as MessagePart;
}

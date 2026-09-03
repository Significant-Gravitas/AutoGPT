/**
 * End-to-end regression for the copilot's Connect card.
 *
 * Every layer of this path had tests and every layer passed, yet composing
 * them produced a card that could never be completed. So this test drives the
 * whole path with nothing in it stubbed:
 *
 *     ToolChain → SetupRequirementsCard → ChainActionCard → ConnectorRow
 *               → ConnectCredentialDialog → useOAuthConnect → the API
 *
 * The API is MSW rather than a live backend, and the consent screen is the one
 * unavoidable fake — no CI job can have a human approve a real OAuth grant.
 * Everything between those two ends is the real component.
 */

import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useConnectedProvidersStore } from "@/app/(platform)/copilot/connectedProvidersStore";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { ToolChain } from "../ToolChain";

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

const REQUIRED_SCOPE = "repo";
const SESSION_ID = "session-1";

// The scopes the browser actually asked the consent screen for. `null` means
// the login endpoint was never called; `""` means it was called without them.
let requestedScopes: string | null = null;
let savedCredentials: Record<string, unknown>[] = [];

describe("copilot Connect card", () => {
  beforeEach(() => {
    requestedScopes = null;
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
        requestedScopes = new URL(request.url).searchParams.get("scopes") ?? "";
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

  afterEach(() => {
    cleanup();
    useCopilotUIStore.setState({ initialPrompt: null, sentMessageCount: 0 });
    // The auto-send claim is a module singleton keyed by session id, so a
    // later test reusing SESSION_ID would silently dismiss its own card.
    useConnectedProvidersStore.getState().clearSession(SESSION_ID);
  });

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
    expect(await screen.findByText("Connected. Continuing…")).toBeDefined();
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
    expect(
      await screen.findByRole("button", { name: "Connect" }),
    ).toBeDefined();
    expect(onSend).not.toHaveBeenCalled();
  });
});

function renderChain() {
  const onSend = vi.fn();
  const utils = render(
    <CredentialsProvider>
      <CopilotChatActionsProvider onSend={onSend}>
        <ToolChain parts={[setupRequirementsPart()]} isStreaming={false} />
      </CopilotChatActionsProvider>
    </CredentialsProvider>,
  );
  return { onSend, ...utils };
}

async function completeConnectFlow() {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: "Connect" }));
  await user.click(await screen.findByText("OAuth"));
  await user.click(await screen.findByRole("button", { name: "Continue" }));
}

/** What the backend returns when a block needs a credential the user's
 *  connected account doesn't cover — here GitHub without the `repo` scope. */
function setupRequirementsPart(): MessagePart {
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
              types: ["oauth2"],
              scopes: [REQUIRED_SCOPE],
            },
          },
        },
      },
    },
  } as unknown as MessagePart;
}

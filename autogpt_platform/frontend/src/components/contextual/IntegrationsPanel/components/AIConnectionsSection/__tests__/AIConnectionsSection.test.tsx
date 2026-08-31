import {
  getGetV2ListChatTransportsMockHandler200,
  getPutV2SetDefaultChatTransportMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getDeleteV1DeleteCredentialsMockHandler200,
  getDeleteV1DeleteCredentialsMockHandler401,
  getGetV1CodexAccountMockHandler200,
  getGetV1ListCredentialsMockHandler200,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AIConnectionsSection } from "../AIConnectionsSection";

function platform(isDefault: boolean): ChatTransportResponse {
  return {
    auth_provider: "platform",
    credential_id: null,
    label: "Self-hosted chat",
    available: true,
    default: isDefault,
  };
}

function chatgpt(isDefault: boolean, id = "cred-1"): ChatTransportResponse {
  return {
    auth_provider: "codex",
    credential_id: id,
    label: "ChatGPT",
    available: true,
    default: isDefault,
  };
}

function mockTransports(
  transports: ChatTransportResponse[],
  credentials: { id: string; username: string }[] = [],
) {
  server.use(
    getGetV2ListChatTransportsMockHandler200({ transports }),
    getPutV2SetDefaultChatTransportMockHandler200({ transports }),
    getGetV1ListCredentialsMockHandler200(
      credentials.map((credential) => ({
        id: credential.id,
        provider: "codex",
        type: "oauth2" as const,
        title: "ChatGPT for Codex",
        scopes: [],
        username: credential.username,
      })),
    ),
  );
}

describe("AIConnectionsSection", () => {
  it("lists every connection with what backs a run on it", async () => {
    mockTransports([platform(true), chatgpt(false)]);

    render(<AIConnectionsSection />);

    expect(await screen.findByText("Self-hosted chat")).toBeDefined();
    expect(screen.getByText("ChatGPT")).toBeDefined();
    expect(
      screen.getByText(/backed by your ChatGPT plan, and spend no AutoGPT/i),
    ).toBeDefined();
  });

  it("names the account a connection runs as", async () => {
    mockTransports(
      [platform(false), chatgpt(true)],
      [{ id: "cred-1", username: "nick@example.com" }],
    );

    render(<AIConnectionsSection />);

    expect(await screen.findByText("nick@example.com")).toBeDefined();
  });

  it("treats a disconnected account snapshot as requiring reconnect", async () => {
    mockTransports(
      [platform(false), chatgpt(true)],
      [{ id: "cred-1", username: "stored@example.com" }],
    );
    server.use(
      getGetV1CodexAccountMockHandler200({
        connected: false,
        requires_openai_auth: true,
        email: "stale@example.com",
        plan_type: "Plus",
      }),
    );

    render(<AIConnectionsSection />);
    await userEvent.click(
      await screen.findByRole("button", { name: "Manage" }),
    );

    expect(
      await screen.findByText(/connection needs to be reconnected/i),
    ).toBeDefined();
    expect(screen.getAllByText("stored@example.com")).toHaveLength(2);
    expect(screen.queryByText("stale@example.com")).toBeNull();
    expect(screen.queryByText("Plus plan")).toBeNull();
  });

  it("keeps the manage dialog open when disconnect fails", async () => {
    mockTransports([platform(false), chatgpt(true)]);
    server.use(
      getGetV1CodexAccountMockHandler200({
        connected: true,
        requires_openai_auth: false,
      }),
      getDeleteV1DeleteCredentialsMockHandler401({ detail: "Not authorized" }),
    );

    render(<AIConnectionsSection />);
    await userEvent.click(
      await screen.findByRole("button", { name: "Manage" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Disconnect" }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Disconnect" })).toBeDefined(),
    );
  });

  it("asks before force-removing an in-use connection", async () => {
    mockTransports([platform(false), chatgpt(true)]);
    server.use(
      getGetV1CodexAccountMockHandler200({
        connected: true,
        requires_openai_auth: false,
      }),
      getDeleteV1DeleteCredentialsMockHandler200(({ request }) =>
        new URL(request.url).searchParams.get("force") === "true"
          ? { deleted: true, revoked: true }
          : {
              deleted: false,
              need_confirmation: true,
              message: "Used by an active schedule",
            },
      ),
    );

    render(<AIConnectionsSection />);
    await userEvent.click(
      await screen.findByRole("button", { name: "Manage" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Disconnect" }));

    expect(
      await screen.findByRole("button", { name: "Force remove" }),
    ).toBeDefined();
    expect(await screen.findByText("Used by an active schedule")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Force remove" }));

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Disconnect" })).toBeNull(),
    );
  });

  it("marks the saved default, and only that one", async () => {
    mockTransports([platform(false), chatgpt(true)]);

    render(<AIConnectionsSection />);

    const options = await screen.findAllByRole("radio");
    expect(options).toHaveLength(2);
    const checked = options.filter(
      (option) => option.getAttribute("aria-checked") === "true",
    );
    expect(checked).toHaveLength(1);
    expect(checked[0].textContent).toContain("ChatGPT");
  });

  it("saves the connection the user picks", async () => {
    mockTransports([platform(true), chatgpt(false)]);
    const saved = vi.fn();
    server.events.on("request:start", ({ request }) => {
      if (
        request.method === "PUT" &&
        request.url.includes("transports/default")
      )
        saved();
    });

    render(<AIConnectionsSection />);
    const chatgptOption = await screen.findByRole("radio", { name: /ChatGPT/ });
    await userEvent.click(chatgptOption);

    await waitFor(() => expect(saved).toHaveBeenCalled());
  });

  it("does not re-save the connection that is already the default", async () => {
    mockTransports([platform(true), chatgpt(false)]);
    const saved = vi.fn();
    server.events.on("request:start", ({ request }) => {
      if (
        request.method === "PUT" &&
        request.url.includes("transports/default")
      )
        saved();
    });

    render(<AIConnectionsSection />);
    const current = await screen.findByRole("radio", {
      name: /Self-hosted chat/,
    });
    await userEvent.click(current);

    await waitFor(() => expect(saved).not.toHaveBeenCalled());
  });

  it("presents no choice when there is only one connection", async () => {
    mockTransports([platform(true)]);

    render(<AIConnectionsSection />);

    // The row still says what powers a chat — it just isn't a decision.
    expect(await screen.findByText("Self-hosted chat")).toBeDefined();
    expect(screen.queryByRole("radiogroup")).toBeNull();
    expect(screen.queryByRole("radio")).toBeNull();
  });

  it("names what is coming without claiming it works", async () => {
    mockTransports([platform(true)]);

    render(<AIConnectionsSection />);

    expect(await screen.findByText("GitHub Copilot and Grok")).toBeDefined();
    expect(screen.getByText("Coming soon")).toBeDefined();
  });
});

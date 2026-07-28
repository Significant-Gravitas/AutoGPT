import { describe, expect, it } from "vitest";

import type { ToolUIPart } from "ai";
import { render, screen } from "@/tests/integrations/test-utils";
import { ListCredentialsTool } from "../ListCredentialsTool";

function makePart(overrides: Record<string, unknown> = {}): ToolUIPart {
  return {
    type: "tool-list_user_credentials",
    toolCallId: "call-list-creds-1",
    state: "input-streaming",
    input: {},
    ...overrides,
  } as ToolUIPart;
}

function makeListOutput() {
  return JSON.stringify({
    type: "credential_list",
    message: "The user has 2 connected credential(s) across 2 provider(s).",
    credentials: [
      {
        id: "cred-1",
        provider: "github",
        type: "oauth2",
        title: "GitHub",
        username: "octocat",
        scopes: ["repo", "read:org"],
        is_managed: false,
      },
      {
        id: "cred-2",
        provider: "notion",
        type: "api_key",
        title: "My Notion key",
        is_managed: false,
      },
    ],
    providers: ["github", "notion"],
    count: 2,
  });
}

describe("ListCredentialsTool", () => {
  it("shows a loading label while streaming", () => {
    const { container } = render(<ListCredentialsTool part={makePart()} />);
    const normalized = (container.textContent ?? "").replace(/\u00a0/g, " ");
    expect(normalized).toContain("Checking connected integrations");
  });

  it("renders connected credentials with provider, type, and account details", () => {
    render(
      <ListCredentialsTool
        part={makePart({ state: "output-available", output: makeListOutput() })}
      />,
    );

    expect(screen.getByText("2 connected integrations")).not.toBeNull();
    expect(screen.getByText("Github")).not.toBeNull();
    expect(screen.getByText("Notion")).not.toBeNull();
    expect(screen.getByText("octocat")).not.toBeNull();
    expect(screen.getByText(/repo, read:org/)).not.toBeNull();
    expect(screen.getByText(/API key · My Notion key/)).not.toBeNull();
  });

  it("shows an empty state when no integrations are connected", () => {
    render(
      <ListCredentialsTool
        part={makePart({
          state: "output-available",
          output: JSON.stringify({
            type: "credential_list",
            message: "The user has not connected any integrations yet.",
            credentials: [],
            providers: [],
            count: 0,
          }),
        })}
      />,
    );

    expect(screen.getByText("No connected integrations")).not.toBeNull();
    expect(
      screen.getAllByText(/has not connected any integrations/).length,
    ).toBeGreaterThan(0);
  });

  it("renders managed and host-scoped credentials with their labels", () => {
    render(
      <ListCredentialsTool
        part={makePart({
          state: "output-available",
          output: JSON.stringify({
            type: "credential_list",
            message: "The user has 2 connected credential(s).",
            credentials: [
              {
                id: "cred-3",
                provider: "agent_mail",
                type: "api_key",
                is_managed: true,
              },
              {
                id: "cred-4",
                provider: "http",
                type: "host_scoped",
                host: "api.example.com",
              },
            ],
            providers: ["agent_mail", "http"],
            count: 2,
          }),
        })}
      />,
    );

    expect(screen.getByText("Agent Mail")).not.toBeNull();
    expect(screen.getByText(/Managed by AutoGPT/)).not.toBeNull();
    expect(screen.getByText(/Host-scoped · api\.example\.com/)).not.toBeNull();
  });

  it("styles the label as an error when the tool returns need_login", () => {
    const { container } = render(
      <ListCredentialsTool
        part={makePart({
          state: "output-available",
          output: JSON.stringify({
            type: "need_login",
            message: "Please sign in to use list_user_credentials",
          }),
        })}
      />,
    );
    const normalized = (container.textContent ?? "").replace(/\u00a0/g, " ");
    expect(normalized).toContain("Could not check connected integrations");
  });

  it("falls back to a done label when output is unparseable", () => {
    const { container } = render(
      <ListCredentialsTool
        part={makePart({
          state: "output-available",
          output: '{"type":"credential_list"',
        })}
      />,
    );
    const normalized = (container.textContent ?? "").replace(/\u00a0/g, " ");
    expect(normalized).toContain("Done");
  });

  it("styles the label as an error on tool failure", () => {
    const { container } = render(
      <ListCredentialsTool
        part={makePart({
          state: "output-error",
          output: '{"type":"error","message":"boom"}',
        })}
      />,
    );
    const normalized = (container.textContent ?? "").replace(/\u00a0/g, " ");
    expect(normalized).toContain("Could not check connected integrations");
  });

  it("surfaces the returned message in an error card", () => {
    render(
      <ListCredentialsTool
        part={makePart({
          state: "output-available",
          output: JSON.stringify({
            type: "error",
            message: "Could not retrieve the user's connected credentials.",
            error: "credential_lookup_failed",
          }),
        })}
      />,
    );

    expect(
      screen.getByText("Could not retrieve the user's connected credentials."),
    ).not.toBeNull();
  });

  it("does not crash on a credential_list payload missing its credentials array", () => {
    const { container } = render(
      <ListCredentialsTool
        part={makePart({
          state: "output-available",
          output: '{"type":"credential_list","count":3}',
        })}
      />,
    );
    const normalized = (container.textContent ?? "").replace(/\u00a0/g, " ");
    expect(normalized).toContain("Done");
    expect(screen.queryByText(/connected integration/)).toBeNull();
  });
});

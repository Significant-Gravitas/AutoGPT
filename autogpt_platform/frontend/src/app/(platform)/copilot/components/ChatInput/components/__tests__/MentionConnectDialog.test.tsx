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
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { CredentialMentionEditor } from "../../../CredentialMention/CredentialMentionEditor";
import { useChatMentions } from "../../useChatMentions";
import { useConnectedIntegrations } from "../../useConnectedIntegrations";
import { MentionConnectDialog } from "../MentionConnectDialog";
import { MentionDropdown } from "../MentionDropdown";

const SENTRY_CREDENTIAL = {
  id: "cred-sentry",
  provider: "mcp",
  type: "oauth2",
  title: "MCP: mcp.sentry.dev",
  username: null,
  scopes: null,
  service: "sentry",
  service_name: "Sentry",
};

interface Props {
  expertId?: string;
  onSend: (message: string) => void;
}

function Composer({ expertId, onSend }: Props) {
  const [value, setValue] = useState("");
  const integrations = useConnectedIntegrations(expertId);
  const mentions = useChatMentions({
    enabled: true,
    value,
    setValue,
    integrations,
    expertId,
    includeWorkspaceFiles: false,
    addWorkspaceFile: vi.fn(),
    addWorkspaceFolder: vi.fn(),
  });
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSend(value);
      }}
    >
      {mentions.isOpen && (
        <MentionDropdown
          {...mentions}
          onSelect={mentions.accept}
          onHighlight={mentions.setHighlightedIndex}
        />
      )}
      <CredentialMentionEditor
        value={value}
        onInputReady={mentions.bindInput}
        onChange={(next, input) => {
          setValue(next);
          mentions.detect(input);
        }}
        onKeyDown={(event) => {
          mentions.onKeyDown(event);
        }}
        onBlur={mentions.close}
        placeholder="Message"
      />
      <MentionConnectDialog
        request={mentions.connectRequest}
        expertId={expertId}
        expertName="Maria"
        onCancel={mentions.cancelConnect}
        onConnected={mentions.resumeAfterConnect}
      />
    </form>
  );
}

/** Sentry connects with a pasted token, so the whole connect runs in-page. */
function arrangeSentry() {
  const state = { connected: false, granted: [] as string[] };
  server.use(
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json(state.connected ? [SENTRY_CREDENTIAL] : []),
    ),
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json([
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
        {
          name: "github",
          display_name: "GitHub",
          supported_auth_types: ["oauth2"],
        },
      ]),
    ),
    http.post("*/api/mcp/discover-tools", () =>
      HttpResponse.json({
        tools: [],
        server_url: "https://mcp.sentry.dev/mcp",
      }),
    ),
    http.post("*/api/mcp/token", () => {
      state.connected = true;
      return HttpResponse.json(SENTRY_CREDENTIAL);
    }),
    http.get("*/api/experts/expert-maria/credentials", () =>
      HttpResponse.json(
        state.granted.includes("cred-sentry")
          ? [
              {
                credential_id: "cred-sentry",
                provider: "mcp",
                title: "MCP: mcp.sentry.dev",
                type: "oauth2",
              },
            ]
          : [],
      ),
    ),
    http.post("*/api/experts/expert-maria/credentials", async ({ request }) => {
      state.granted = (
        (await request.json()) as { credential_ids: string[] }
      ).credential_ids;
      return HttpResponse.json([]);
    }),
  );
  return state;
}

async function openConnectFromMention() {
  await userEvent.type(screen.getByRole("textbox"), "Check @sentry");
  await screen.findByText("No connected integrations to mention.");
  await userEvent.keyboard("{ArrowDown}{Enter}");
  return screen.findByRole("dialog", { name: "Connect a service" });
}

async function connectSentry(dialog: HTMLElement) {
  fireEvent.click(
    await within(dialog).findByRole("button", { name: /Sentry/ }),
  );
  fireEvent.change(await within(dialog).findByLabelText("API token"), {
    target: { value: "sntrys_token" },
  });
  fireEvent.click(within(dialog).getByRole("button", { name: "Save token" }));
}

describe("Connect an integration from the @ mention list", () => {
  it("opens the connect dialog searched for what was typed", async () => {
    arrangeSentry();
    render(<Composer onSend={vi.fn()} />);

    const dialog = await openConnectFromMention();

    expect(
      within(dialog).getByRole("textbox", { name: "Search services" }),
    ).toHaveProperty("value", "sentry");
    expect(
      await within(dialog).findByRole("button", { name: /Sentry/ }),
    ).toBeDefined();
    expect(within(dialog).queryByRole("button", { name: /GitHub/ })).toBeNull();
  });

  it("lists the new account under the same @query once connected, without sending", async () => {
    arrangeSentry();
    const onSend = vi.fn();
    render(<Composer onSend={onSend} />);

    await connectSentry(await openConnectFromMention());

    expect(
      await screen.findByRole("option", { name: /mcp\.sentry\.dev/ }),
    ).toBeDefined();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    const textbox = screen.getByRole("textbox");
    expect(textbox).toHaveProperty("value", "Check @sentry");
    await waitFor(() => expect(document.activeElement).toBe(textbox));
    expect(textbox).toHaveProperty("selectionStart", 13);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("offers the new account even when the service was found by its description", async () => {
    arrangeSentry();
    render(<Composer onSend={vi.fn()} />);

    await userEvent.type(screen.getByRole("textbox"), "Check @errors");
    await screen.findByText("No connected integrations to mention.");
    await userEvent.keyboard("{ArrowDown}{Enter}");
    await connectSentry(
      await screen.findByRole("dialog", { name: "Connect a service" }),
    );

    expect(
      await screen.findByRole("option", { name: /mcp\.sentry\.dev/ }),
    ).toBeDefined();
  });

  it("gives the chat's expert the new account, so its @ list offers it", async () => {
    const state = arrangeSentry();
    render(<Composer expertId="expert-maria" onSend={vi.fn()} />);

    await connectSentry(await openConnectFromMention());

    await waitFor(() => expect(state.granted).toEqual(["cred-sentry"]));
    expect(
      await screen.findByRole("option", { name: /mcp\.sentry\.dev/ }),
    ).toBeDefined();
  });

  it("saving an API key in the dialog does not send the half-written message", async () => {
    let connected = false;
    server.use(
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(
          connected
            ? [
                {
                  id: "cred-linear",
                  provider: "linear",
                  type: "api_key",
                  title: "Team Linear",
                  username: null,
                  scopes: null,
                },
              ]
            : [],
        ),
      ),
      http.get("*/api/integrations/providers", () =>
        HttpResponse.json([
          {
            name: "linear",
            display_name: "Linear",
            supported_auth_types: ["api_key"],
          },
        ]),
      ),
      http.post("*/api/integrations/linear/credentials", () => {
        connected = true;
        return HttpResponse.json({
          id: "cred-linear",
          provider: "linear",
          type: "api_key",
          title: "Team Linear",
        });
      }),
    );
    const onSend = vi.fn();
    render(<Composer onSend={onSend} />);

    await userEvent.type(screen.getByRole("textbox"), "Check @linear");
    await screen.findByText("No connected integrations to mention.");
    await userEvent.keyboard("{ArrowDown}{Enter}");
    const dialog = await screen.findByRole("dialog", {
      name: "Connect a service",
    });
    fireEvent.click(
      await within(dialog).findByRole("button", { name: /Linear/ }),
    );
    await userEvent.type(
      await within(dialog).findByPlaceholderText("My Linear key"),
      "Team Linear",
    );
    await userEvent.type(
      within(dialog).getByPlaceholderText("sk-..."),
      "lin_api_key",
    );
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Save API key" }),
    );

    expect(
      await screen.findByRole("option", { name: /Team Linear/ }),
    ).toBeDefined();
    expect(onSend).not.toHaveBeenCalled();
  });

  it("closing the dialog leaves the message as typed, with the caret back after it", async () => {
    arrangeSentry();
    const onSend = vi.fn();
    render(<Composer onSend={onSend} />);

    const dialog = await openConnectFromMention();
    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    const textbox = screen.getByRole("textbox");
    expect(textbox).toHaveProperty("value", "Check @sentry");
    await waitFor(() => expect(document.activeElement).toBe(textbox));
    expect(textbox).toHaveProperty("selectionStart", 13);
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(onSend).not.toHaveBeenCalled();
  });
});

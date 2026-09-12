import { beforeEach, describe, expect, test } from "vitest";
import { http, HttpResponse } from "msw";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getGetV1ListProvidersMockHandler } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import SettingsIntegrationsPage from "../page";
import {
  authProviders,
  discoveryRequest,
  oauthRequest,
  setupAuthFixtures,
  tokenRequest,
} from "./mcp-auth-fixtures";

beforeEach(() => {
  setupAuthFixtures();
  server.use(
    getGetV1ListProvidersMockHandler([
      ...authProviders,
      {
        name: "github",
        description: "Issues and PRs",
        supported_auth_types: ["api_key"],
      },
      {
        name: "mcp",
        description: "Connect any MCP server",
        supported_auth_types: ["oauth2"],
      },
    ]),
  );
});

async function openPicker() {
  const buttons = await screen.findAllByRole("button", {
    name: /connect.*service/i,
  });
  fireEvent.click(buttons[0]);
  return screen.findByRole("dialog");
}

describe("SettingsIntegrationsPage — MCP catalogue", () => {
  test("lists native providers and branded MCP entries together", async () => {
    render(<SettingsIntegrationsPage />);
    expect(
      await screen.findByRole("button", { name: /github.*issues and prs/i }),
    ).toBeDefined();
    const preset = await screen.findByRole("button", {
      name: /agentmail.*account/i,
    });
    expect(within(preset).getByText("MCP", { exact: true })).toBeDefined();
    expect(preset.querySelector("img")?.getAttribute("src")).toBe(
      "/integrations/mcp.png",
    );
    expect(screen.queryByText("Mcp Agentmail")).toBeNull();
  });

  test("search filters native and MCP providers by display name", async () => {
    render(<SettingsIntegrationsPage />);
    await screen.findByRole("button", { name: /agentmail.*account/i });
    fireEvent.change(screen.getByLabelText(/search integrations/i), {
      target: { value: "agentmail" },
    });
    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: /github.*issues and prs/i }),
      ).toBeNull();
    });
    expect(
      screen.getByRole("button", { name: /agentmail.*account/i }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: /intercom.*account/i }),
    ).toBeNull();
  });

  test("a page preset locks its URL and uses generic MCP OAuth", async () => {
    render(<SettingsIntegrationsPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: /agentmail.*account/i }),
    );
    const dialog = await screen.findByRole("dialog");
    const input =
      await within(dialog).findByLabelText<HTMLInputElement>("Server URL");
    expect(input.value).toBe("https://mcp.agentmail.example.com/mcp");
    expect(input.readOnly).toBe(true);
    expect(
      within(dialog).getByAltText("AgentMail logo").getAttribute("src"),
    ).toBe("/integrations/mcp.png");
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    await waitFor(() => {
      expect(oauthRequest).toHaveBeenCalledWith({
        server_url: "https://mcp.agentmail.example.com/mcp",
      });
    });
  });

  test("the service picker opens the same named MCP presets", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPicker();
    fireEvent.click(
      await within(dialog).findByRole("button", {
        name: /agentmail.*account/i,
      }),
    );
    expect(
      (await within(dialog).findByLabelText<HTMLInputElement>("Server URL"))
        .value,
    ).toBe("https://mcp.agentmail.example.com/mcp");
  });

  test("generic MCP setup retains a blank editable URL", async () => {
    render(<SettingsIntegrationsPage />);
    const dialog = await openPicker();
    fireEvent.click(await within(dialog).findByText("Connect any MCP server"));
    const input =
      await within(dialog).findByLabelText<HTMLInputElement>("Server URL");
    expect(input.value).toBe("");
    expect(input.readOnly).toBe(false);
    expect(
      within(dialog).getByRole<HTMLButtonElement>("button", {
        name: /^connect$/i,
      }).disabled,
    ).toBe(true);
  });

  test.each([0, 2])(
    "public discovery reports %i tools without saving or reusing credentials",
    async (toolCount) => {
      server.use(
        http.post("*/api/mcp/discover-tools", async ({ request }) => {
          discoveryRequest(await request.json());
          return HttpResponse.json({
            tools: Array.from({ length: toolCount }, (_, i) => ({
              name: "tool_" + i,
              description: "Tool",
              input_schema: {},
            })),
          });
        }),
      );
      render(<SettingsIntegrationsPage />);
      fireEvent.click(
        await screen.findByRole("button", { name: /parallel.*account/i }),
      );
      const dialog = await screen.findByRole("dialog");
      fireEvent.click(
        await within(dialog).findByRole("button", {
          name: /check connection/i,
        }),
      );
      expect((await within(dialog).findByRole("status")).textContent).toBe(
        toolCount
          ? "2 tools available."
          : "Connected, but this server returned no tools.",
      );
      expect(
        within(dialog).getByText(/no connection was saved/i),
      ).toBeDefined();
      expect(within(dialog).getByText(/use this server url/i)).toBeDefined();
      expect(discoveryRequest).toHaveBeenCalledWith({
        server_url: "https://search.parallel.ai/mcp",
        use_saved_credentials: false,
      });
      expect(tokenRequest).not.toHaveBeenCalled();
      expect(oauthRequest).not.toHaveBeenCalled();
    },
  );
});

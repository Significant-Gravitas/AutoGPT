import { beforeEach, describe, expect, test } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import SettingsIntegrationsPage from "../page";
import {
  discoveryRequest,
  oauthRequest,
  setupAuthFixtures,
  tokenRequest,
} from "./mcp-auth-fixtures";

beforeEach(setupAuthFixtures);

async function openPreset(name: string) {
  render(<SettingsIntegrationsPage />);
  fireEvent.click(
    await screen.findByRole("button", { name: new RegExp(name + ".*account") }),
  );
  const dialog = await screen.findByRole("dialog");
  await within(dialog).findByLabelText("Server URL");
  return dialog;
}

describe("MCP preset authentication methods", () => {
  test("AgentMail never offers a manual credential, including after OAuth failure", async () => {
    const dialog = await openPreset("AgentMail");
    expect(within(dialog).queryByText(/use an api token instead/i)).toBeNull();
    expect(within(dialog).queryByLabelText("Authentication type")).toBeNull();
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    expect(await within(dialog).findByRole("alert")).toBeDefined();
    expect(within(dialog).queryByPlaceholderText("Paste API token")).toBeNull();
    expect(
      within(dialog).queryByRole("button", { name: /save token/i }),
    ).toBeNull();
    expect(oauthRequest).toHaveBeenCalledTimes(1);
  });

  test("Langfuse defaults to Basic and rejects a pasted Bearer scheme before probing", async () => {
    const dialog = await openPreset("Langfuse");
    const url = within(dialog).getByLabelText<HTMLInputElement>("Server URL");
    expect(url.readOnly).toBe(false);
    expect(url.value).toBe("");
    const region = within(dialog).getByRole("combobox", {
      name: "Server region",
    });
    fireEvent.keyDown(region, { key: "ArrowDown" });
    fireEvent.click(await screen.findByRole("option", { name: "EU" }));
    expect(url.value).toBe("https://cloud.langfuse.com/api/public/mcp");
    expect(url.readOnly).toBe(true);
    fireEvent.change(within(dialog).getByPlaceholderText(/paste base64/i), {
      target: { value: "secret-for-old-region" },
    });
    fireEvent.keyDown(region, { key: "ArrowDown" });
    fireEvent.click(await screen.findByRole("option", { name: "Custom URL" }));
    expect(url.value).toBe("");
    expect(url.readOnly).toBe(false);
    expect(
      within(dialog).getByPlaceholderText<HTMLInputElement>(/paste base64/i)
        .value,
    ).toBe("");
    fireEvent.change(url, {
      target: { value: "https://cloud.langfuse.com/api/public/mcp" },
    });
    expect(within(dialog).queryByLabelText("Authentication type")).toBeNull();
    expect(
      within(dialog).queryByRole("button", { name: /try oauth/i }),
    ).toBeNull();
    const token = within(dialog).getByPlaceholderText(/paste base64/i);
    fireEvent.change(token, {
      target: { value: "Bearer cHVibGljOnNlY3JldA==" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    expect(await within(dialog).findByRole("alert")).toBeDefined();
    expect(discoveryRequest).not.toHaveBeenCalled();
    expect(tokenRequest).not.toHaveBeenCalled();
    fireEvent.change(token, { target: { value: "cHVibGljOnNlY3JldA==" } });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    await waitFor(() =>
      expect(tokenRequest).toHaveBeenCalledWith({
        server_url: "https://cloud.langfuse.com/api/public/mcp",
        token: "Basic cHVibGljOnNlY3JldA==",
      }),
    );
    expect(discoveryRequest.mock.invocationCallOrder[0]).toBeLessThan(
      tokenRequest.mock.invocationCallOrder[0],
    );
    expect(oauthRequest).not.toHaveBeenCalled();
  });

  test("Intercom offers only Bearer and rejects an explicit Basic credential", async () => {
    const dialog = await openPreset("Intercom");
    expect(
      within(dialog).queryByRole("button", { name: /try oauth/i }),
    ).toBeNull();
    expect(within(dialog).queryByLabelText("Authentication type")).toBeNull();
    const token = within(dialog).getByPlaceholderText("Paste API token");
    fireEvent.change(token, { target: { value: "Basic dXNlcjpwYXNz" } });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    expect(await within(dialog).findByRole("alert")).toBeDefined();
    expect(discoveryRequest).not.toHaveBeenCalled();
    fireEvent.change(token, { target: { value: "intercom-access-token" } });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    await waitFor(() =>
      expect(tokenRequest).toHaveBeenCalledWith({
        server_url: "https://mcp.intercom.example.com/mcp",
        token: "Bearer intercom-access-token",
      }),
    );
    expect(oauthRequest).not.toHaveBeenCalled();
  });
});

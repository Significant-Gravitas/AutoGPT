import { beforeEach, describe, expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";
import { server } from "@/mocks/mock-server";
import { openOAuthPopup } from "@/lib/oauth-popup";
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

vi.mock("@/lib/oauth-popup", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/oauth-popup")>()),
  openOAuthPopup: vi.fn(),
}));

async function openPreset(name: string) {
  render(<SettingsIntegrationsPage />);
  fireEvent.click(
    await screen.findByRole("button", { name: new RegExp(name + ".*account") }),
  );
  const dialog = await screen.findByRole("dialog");
  await within(dialog).findByLabelText("Server URL");
  return dialog;
}

describe("MCP preset access options", () => {
  test("switching preset methods cancels sign-in and discards entered credentials", async () => {
    server.use(
      http.post("*/api/mcp/oauth/login", () =>
        HttpResponse.json({
          login_url: "https://vendor.example.com/authorize",
          state_token: "state",
        }),
      ),
    );
    const controller = new AbortController();
    const abort = vi.fn(() => controller.abort());
    vi.mocked(openOAuthPopup).mockImplementation(() => ({
      promise: new Promise((_resolve, reject) =>
        controller.signal.addEventListener("abort", () =>
          reject(new Error("OAuth flow was canceled")),
        ),
      ),
      cleanup: { abort, signal: controller.signal },
      popupBlocked: false,
      fallbackBlocked: false,
    }));
    const dialog = await openPreset("Parallel");
    fireEvent.mouseDown(
      within(dialog).getByRole("tab", { name: /^sign in$/i }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    await waitFor(() => expect(openOAuthPopup).toHaveBeenCalled());
    fireEvent.mouseDown(
      within(dialog).getByRole("tab", { name: /^api token$/i }),
      { button: 0, ctrlKey: false },
    );
    expect(abort).toHaveBeenCalledTimes(1);
    fireEvent.change(within(dialog).getByPlaceholderText("Paste API token"), {
      target: { value: "unsaved-secret" },
    });
    fireEvent.mouseDown(
      within(dialog).getByRole("tab", { name: /^no sign-in$/i }),
      { button: 0, ctrlKey: false },
    );
    fireEvent.mouseDown(
      within(dialog).getByRole("tab", { name: /^api token$/i }),
      { button: 0, ctrlKey: false },
    );
    expect(
      within(dialog).getByPlaceholderText<HTMLInputElement>("Paste API token")
        .value,
    ).toBe("");
    expect(discoveryRequest).not.toHaveBeenCalled();
    expect(tokenRequest).not.toHaveBeenCalled();
  });

  test("Parallel defaults public, offers OAuth on its distinct URL, and saves tokens on the base URL", async () => {
    server.use(
      http.post("*/api/mcp/oauth/login", async ({ request }) => {
        oauthRequest(await request.json());
        return HttpResponse.json({
          login_url: "https://vendor.example.com/authorize",
          state_token: "state",
        });
      }),
    );
    vi.mocked(openOAuthPopup).mockImplementation(() => ({
      promise: Promise.reject(new Error("Vendor rejected this sign-in")),
      cleanup: { abort: vi.fn(), signal: new AbortController().signal },
      popupBlocked: false,
      fallbackBlocked: false,
    }));
    const dialog = await openPreset("Parallel");
    fireEvent.click(
      within(dialog).getByRole("button", { name: /check connection/i }),
    );
    await waitFor(() =>
      expect(discoveryRequest).toHaveBeenCalledWith({
        server_url: "https://search.parallel.ai/mcp",
        use_saved_credentials: false,
      }),
    );
    expect(tokenRequest).not.toHaveBeenCalled();
    expect(oauthRequest).not.toHaveBeenCalled();
    fireEvent.mouseDown(
      within(dialog).getByRole("tab", { name: /^sign in$/i }),
      { button: 0, ctrlKey: false },
    );
    expect(
      within(dialog).getByLabelText<HTMLInputElement>("Server URL").value,
    ).toBe("https://search.parallel.ai/mcp-oauth");
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    await waitFor(() =>
      expect(oauthRequest).toHaveBeenCalledWith({
        server_url: "https://search.parallel.ai/mcp-oauth",
      }),
    );
    expect((await within(dialog).findByRole("alert")).textContent).toBe(
      "Vendor rejected this sign-in",
    );
    fireEvent.mouseDown(
      within(dialog).getByRole("tab", { name: /^api token$/i }),
      { button: 0, ctrlKey: false },
    );
    expect(
      within(dialog).getByLabelText<HTMLInputElement>("Server URL").value,
    ).toBe("https://search.parallel.ai/mcp");
    fireEvent.change(within(dialog).getByPlaceholderText("Paste API token"), {
      target: { value: "parallel-access-token" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    await waitFor(() =>
      expect(tokenRequest).toHaveBeenCalledWith({
        server_url: "https://search.parallel.ai/mcp",
        token: "Bearer parallel-access-token",
      }),
    );
    expect(discoveryRequest).toHaveBeenCalledWith({
      server_url: "https://search.parallel.ai/mcp",
      auth_token: "Bearer parallel-access-token",
    });
  });

  test("Customer.io defaults to read access and adds only documented draft writes when selected", async () => {
    const dialog = await openPreset("Customer.io");
    const changes = within(dialog).getByRole("checkbox", {
      name: /allow changes/i,
    });
    expect(changes.getAttribute("aria-checked")).toBe("false");
    expect(within(dialog).getByText(/create drafts/i)).toBeDefined();
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    await waitFor(() =>
      expect(oauthRequest).toHaveBeenLastCalledWith({
        server_url: "https://mcp.customer_io.example.com/mcp",
        scopes: ["read"],
      }),
    );
    await within(dialog).findByRole("alert");
    fireEvent.click(changes);
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    await waitFor(() =>
      expect(oauthRequest).toHaveBeenLastCalledWith({
        server_url: "https://mcp.customer_io.example.com/mcp",
        scopes: ["read", "write"],
      }),
    );
    expect(tokenRequest).not.toHaveBeenCalled();
  });
});

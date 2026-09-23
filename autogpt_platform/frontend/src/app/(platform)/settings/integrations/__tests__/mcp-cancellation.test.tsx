import { beforeEach, describe, expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { openOAuthPopup } from "@/lib/oauth-popup";
import SettingsIntegrationsPage from "../page";
import { setupAuthFixtures, tokenRequest } from "./mcp-auth-fixtures";

vi.mock("@/lib/oauth-popup", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/oauth-popup")>()),
  openOAuthPopup: vi.fn(),
}));

beforeEach(setupAuthFixtures);

function delayedResponse() {
  let release = () => {};
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { pending, release };
}

async function openParallel() {
  render(<SettingsIntegrationsPage />);
  fireEvent.click(
    await screen.findByRole("button", { name: /parallel.*account/i }),
  );
  const dialog = await screen.findByRole("dialog");
  await within(dialog).findByLabelText("Server URL");
  return dialog;
}

function selectMethod(dialog: HTMLElement, name: RegExp) {
  fireEvent.mouseDown(within(dialog).getByRole("tab", { name }), {
    button: 0,
    ctrlKey: false,
  });
}

async function releaseResponse(gate: ReturnType<typeof delayedResponse>) {
  await act(async () => {
    gate.release();
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
}

describe("MCP preset cancellation during requests", () => {
  test("a delayed login response cannot open a popup after switching methods", async () => {
    const gate = delayedResponse();
    const started = vi.fn();
    const exchanged = vi.fn();
    const requestSignals: AbortSignal[] = [];
    server.use(
      http.post("*/api/mcp/oauth/login", async ({ request }) => {
        requestSignals.push(request.signal);
        started();
        await gate.pending;
        return HttpResponse.json({
          login_url: "https://vendor.example.com/login",
          state_token: "state",
        });
      }),
      http.post("*/api/mcp/oauth/callback", () => {
        exchanged();
        return HttpResponse.json({});
      }),
    );
    vi.mocked(openOAuthPopup).mockImplementation(() => ({
      promise: new Promise(() => {}),
      cleanup: { abort: vi.fn(), signal: new AbortController().signal },
      popupBlocked: false,
      fallbackBlocked: false,
    }));
    const dialog = await openParallel();
    selectMethod(dialog, /^sign in$/i);
    fireEvent.click(within(dialog).getByRole("button", { name: /^connect$/i }));
    await waitFor(() => expect(started).toHaveBeenCalled());
    selectMethod(dialog, /^api token$/i);
    expect(requestSignals[0].aborted).toBe(true);
    await releaseResponse(gate);
    expect(openOAuthPopup).not.toHaveBeenCalled();
    expect(exchanged).not.toHaveBeenCalled();
    expect(tokenRequest).not.toHaveBeenCalled();
    expect(
      within(dialog).getByPlaceholderText("Paste API token"),
    ).toBeDefined();
  });

  test("a delayed token probe cannot save a credential after switching methods", async () => {
    const gate = delayedResponse();
    const started = vi.fn();
    const requestSignals: AbortSignal[] = [];
    server.use(
      http.post("*/api/mcp/discover-tools", async ({ request }) => {
        requestSignals.push(request.signal);
        started();
        await gate.pending;
        return HttpResponse.json({ tools: [] });
      }),
    );
    const dialog = await openParallel();
    selectMethod(dialog, /^api token$/i);
    fireEvent.change(within(dialog).getByPlaceholderText("Paste API token"), {
      target: { value: "unsaved-secret" },
    });
    fireEvent.click(
      within(dialog).getByRole("button", { name: /save token/i }),
    );
    await waitFor(() => expect(started).toHaveBeenCalled());
    selectMethod(dialog, /^no sign-in$/i);
    expect(requestSignals[0].aborted).toBe(true);
    await releaseResponse(gate);
    expect(tokenRequest).not.toHaveBeenCalled();
    expect(
      within(dialog).getByRole("button", { name: /check connection/i }),
    ).toBeDefined();
  });
});

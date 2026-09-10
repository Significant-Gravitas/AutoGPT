import {
  render,
  screen,
  fireEvent,
  cleanup,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi, beforeEach } from "vitest";
import { AyrshareConnectButton } from "../AyrshareConnectButton";
import { CredentialsActionsContext } from "@/providers/agent-credentials/credentials-provider";

vi.mock("@/app/api/__generated__/endpoints/integrations/integrations", () => ({
  getV1GetAyrshareSsoUrl: vi.fn(),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

import { getV1GetAyrshareSsoUrl } from "@/app/api/__generated__/endpoints/integrations/integrations";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function renderWithActions(reload: () => void) {
  // Stub window.open so the real popup doesn't fire during tests.
  vi.stubGlobal(
    "open",
    vi.fn(() => ({}) as Window),
  );
  return render(
    <CredentialsActionsContext.Provider value={{ reload }}>
      <AyrshareConnectButton />
    </CredentialsActionsContext.Provider>,
  );
}

describe("AyrshareConnectButton", () => {
  beforeEach(() => {
    vi.mocked(getV1GetAyrshareSsoUrl).mockReset();
  });

  it("reloads the credentials context after a successful SSO URL fetch", async () => {
    vi.mocked(getV1GetAyrshareSsoUrl).mockResolvedValue({
      status: 200,
      data: { sso_url: "https://app.ayrshare.com/sso/fake" },
    } as any);
    const reload = vi.fn();

    renderWithActions(reload);
    fireEvent.click(
      screen.getByRole("button", { name: /Connect Social Media Accounts/i }),
    );

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
  });

  it("does NOT reload the credentials context when the SSO URL fetch fails", async () => {
    vi.mocked(getV1GetAyrshareSsoUrl).mockResolvedValue({
      status: 500,
      data: { detail: "boom" },
    } as any);
    const reload = vi.fn();

    renderWithActions(reload);
    fireEvent.click(
      screen.getByRole("button", { name: /Connect Social Media Accounts/i }),
    );

    // Let the handler's promise chain resolve + the finally block run.
    await waitFor(() => expect(getV1GetAyrshareSsoUrl).toHaveBeenCalled());
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(reload).not.toHaveBeenCalled();
  });
});

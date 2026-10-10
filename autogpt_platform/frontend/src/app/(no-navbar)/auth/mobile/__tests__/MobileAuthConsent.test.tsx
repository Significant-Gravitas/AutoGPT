import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MobileAuthConsent } from "../components/MobileAuthConsent";

const CHALLENGE = "A".repeat(43);
const STATE = "B".repeat(43);
const CODE = "C".repeat(43);
const CALLBACK = `autogpt://auth/callback?code=${CODE}&state=${STATE}`;

describe("mobile app sign-in consent", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ url: CALLBACK }),
    });
    vi.spyOn(window.location, "assign").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    fetchMock.mockReset();
  });

  function renderConsent() {
    return render(
      <MobileAuthConsent
        userID="user-123"
        email="tester@agpt.co"
        codeChallenge={CHALLENGE}
        state={STATE}
      />,
    );
  }

  it("shows the account and waits for explicit approval", () => {
    renderConsent();
    expect(screen.getByText("tester@agpt.co")).not.toBeNull();
    expect(
      screen.getByRole("button", { name: "Connect AutoGPT" }),
    ).not.toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(window.location.assign).not.toHaveBeenCalled();
  });

  it("binds consent to the request and returns the one-time code to the app", async () => {
    renderConsent();
    await userEvent.click(
      screen.getByRole("button", { name: "Connect AutoGPT" }),
    );
    expect(fetchMock).toHaveBeenCalledWith("/api/auth/mobile/authorize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({
        code_challenge: CHALLENGE,
        state: STATE,
        expected_user_id: "user-123",
      }),
    });
    await waitFor(() => {
      expect(window.location.assign).toHaveBeenCalledWith(CALLBACK);
    });
  });

  it("offers retry when authorization fails", async () => {
    fetchMock.mockResolvedValueOnce({ ok: false });
    renderConsent();
    await userEvent.click(
      screen.getByRole("button", { name: "Connect AutoGPT" }),
    );
    expect((await screen.findByRole("alert")).textContent).toContain(
      "Could not connect AutoGPT",
    );
    expect(window.location.assign).not.toHaveBeenCalled();
    expect(
      (
        screen.getByRole("button", {
          name: "Connect AutoGPT",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
  });

  it("returns a state-bound cancellation without authorizing a session", async () => {
    renderConsent();
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(fetchMock).not.toHaveBeenCalled();
    expect(window.location.assign).toHaveBeenCalledWith(
      `autogpt://auth/callback?error=access_denied&state=${STATE}`,
    );
  });

  it.each([
    `https://attacker.example/?code=${CODE}&state=${STATE}`,
    `autogpt://other/callback?code=${CODE}&state=${STATE}`,
    `autogpt://auth/callback?code=${CODE}&state=${"D".repeat(43)}`,
    `autogpt://auth/callback?code=${CODE}&state=${STATE}&token=unexpected`,
  ])("rejects an unexpected callback", async (url) => {
    fetchMock.mockResolvedValueOnce({ ok: true, json: async () => ({ url }) });
    renderConsent();
    await userEvent.click(
      screen.getByRole("button", { name: "Connect AutoGPT" }),
    );
    expect(await screen.findByRole("alert")).not.toBeNull();
    expect(window.location.assign).not.toHaveBeenCalled();
  });
});

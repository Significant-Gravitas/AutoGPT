import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DeviceAuthConnectButton } from "../DeviceAuthConnectButton";

const state = {
  connect: vi.fn(),
  cancel: vi.fn(),
  phase: "idle" as string,
  userCode: "",
  verificationUrl: "",
};

vi.mock("../useDeviceAuthConnect", () => ({
  useDeviceAuthConnect: () => state,
}));

function renderButton() {
  return render(
    <DeviceAuthConnectButton
      provider="stripe_link"
      providerName="Stripe Link"
      onSuccess={vi.fn()}
    />,
  );
}

describe("DeviceAuthConnectButton", () => {
  it("offers a connect action before the flow starts", () => {
    state.phase = "idle";
    state.userCode = "";
    renderButton();

    expect(screen.getByRole("button")).toBeTruthy();
  });

  it("shows the code phrase and the approval link while waiting", () => {
    state.phase = "polling";
    state.userCode = "glow-relish-chaste-soft";
    state.verificationUrl = "https://app.link.com/device/setup?code=glow";
    renderButton();

    expect(screen.getByText("glow-relish-chaste-soft")).toBeTruthy();
    const link = screen.getByRole("link") as HTMLAnchorElement;
    expect(link.href).toContain("app.link.com/device/setup");
  });

  it("keeps the code phrase out of session replay", () => {
    // `Text` adds `sentry-unmask` by default; this is a live authorization
    // code, so it must not be recorded. Regression test for that default.
    state.phase = "polling";
    state.userCode = "glow-relish-chaste-soft";
    renderButton();

    const code = screen.getByText("glow-relish-chaste-soft");
    expect(code.className).not.toContain("sentry-unmask");
  });

  // Between clicking Connect and the initiate call returning there is no code
  // and no URL yet. Rendering the code panel then showed an empty box above an
  // "Open Stripe Link" link with href="" that just reloaded the builder.
  it("shows a starting state instead of an empty code and a dead link", () => {
    state.phase = "awaiting_user";
    state.userCode = "";
    state.verificationUrl = "";
    const { container } = renderButton();

    expect(screen.getByText(/Starting device authorization/)).toBeTruthy();
    expect(container.querySelector("a")).toBeNull();
  });
});

import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../../store";
import { ConnectionPicker } from "../ConnectionPicker/ConnectionPicker";
import {
  mockMaxUpgrade,
  deploymentOffer,
  openPicker,
} from "./maxUpgradeFixtures";

beforeEach(() => {
  useCopilotUIStore.setState({
    copilotLlmAuth: null,
    copilotLlmModel: "standard",
  });
});

describe("contextual Max upsell", () => {
  it("links the locked server model to existing billing without fetching or showing prices", async () => {
    const billingRequest = vi.fn(() => HttpResponse.error());
    mockMaxUpgrade();
    server.use(http.all("*/api/credits/*", billingRequest));
    render(<ConnectionPicker />);
    await openPicker();

    const upgrade = await screen.findByRole("link", { name: "Upgrade to Max" });
    expect(upgrade.getAttribute("href")).toBe("/settings/billing");
    expect(screen.getByText("opus-server").isConnected).toBe(true);
    const group = screen.getByRole("radiogroup", { name: "Model tier" });
    const advanced = within(group).getByRole("radio", {
      name: /Advanced.*opus-server/,
    });
    expect(advanced.getAttribute("aria-disabled")).toBe("true");
    expect(advanced.getAttribute("aria-checked")).toBe("false");
    expect(advanced.getAttribute("tabindex")).toBe("-1");
    expect(
      within(group)
        .getByRole("radio", { name: /Balanced.*sonnet-server/ })
        .getAttribute("aria-checked"),
    ).toBe("true");
    expect(
      screen.queryByText(
        /\$\d|prorat|higher usage|priority support|file storage/i,
      ),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: /Review upgrade|Upgrade to Max/ }),
    ).toBeNull();
    expect(billingRequest).not.toHaveBeenCalled();
    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard");
  });

  it("preserves the ChatGPT connection entry below the Max offer", async () => {
    mockMaxUpgrade();
    render(<ConnectionPicker />);
    await openPicker();

    const upgrade = await screen.findByRole("link", { name: "Upgrade to Max" });
    expect(screen.getByText("Add a connection").isConnected).toBe(true);
    const connect = screen.getByRole("button", {
      name: "Connect a ChatGPT subscription",
    });
    expect(within(connect).getByText("ChatGPT subscription").isConnected).toBe(
      true,
    );
    expect(connect.hasAttribute("disabled")).toBe(false);
    expect(
      upgrade.compareDocumentPosition(connect) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("retains the ChatGPT plan gate alongside the Advanced upsell", async () => {
    mockMaxUpgrade([
      deploymentOffer(),
      deploymentOffer({
        offer_id: "codex:locked",
        provider_family: "openai",
        display_name: "ChatGPT",
        auth_method: "chatgpt_oauth",
        state: "locked",
        selectable: false,
        is_default: false,
        tiers: [],
        lock_reason: "A Max plan or higher is required to use ChatGPT.",
        unlock_href: "/settings/billing",
      }),
    ]);
    render(<ConnectionPicker />);
    await openPicker();

    expect(
      (
        await screen.findByRole("link", { name: "Upgrade to Max" })
      ).getAttribute("href"),
    ).toBe("/settings/billing");
    expect(
      screen.getByText("A Max plan or higher is required to use ChatGPT.")
        .isConnected,
    ).toBe(true);
    expect(
      screen.getByRole("link", { name: "See plans" }).getAttribute("href"),
    ).toBe("/settings/billing");
    expect(
      screen.queryByRole("button", { name: "Connect a ChatGPT subscription" }),
    ).toBeNull();
  });

  it("keeps a connected provider's available Advanced tier selectable", async () => {
    const linked = deploymentOffer({
      offer_id: "codex:cred-1",
      provider_family: "openai",
      display_name: "ChatGPT",
      auth_method: "chatgpt_oauth",
      credential_id: "cred-1",
      backed_by_label: "Your ChatGPT plan",
      is_default: false,
      tiers: [
        {
          tier: "standard",
          label: "Balanced",
          selectable: true,
          display_model: null,
        },
        {
          tier: "advanced",
          label: "Advanced",
          selectable: true,
          display_model: null,
        },
      ],
    });
    mockMaxUpgrade([deploymentOffer(), linked]);
    render(<ConnectionPicker />);
    await openPicker();
    await userEvent.click(
      await screen.findByRole("radio", { name: /ChatGPT/ }),
    );
    await userEvent.click(screen.getByRole("radio", { name: "Advanced" }));

    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("advanced");
    expect(screen.queryByRole("link", { name: "Upgrade to Max" })).toBeNull();
    expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
      authProvider: "codex",
      credentialId: "cred-1",
    });
  });
});

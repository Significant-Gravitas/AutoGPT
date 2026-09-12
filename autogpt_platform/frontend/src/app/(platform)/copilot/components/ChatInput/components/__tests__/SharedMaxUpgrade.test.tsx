import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../../store";
import { ConnectionPicker } from "../ConnectionPicker/ConnectionPicker";
import {
  availableDeploymentOffer,
  deploymentOffer,
  lockedChatGPTOffer,
  mockMaxUpgrade,
  openPicker,
} from "./maxUpgradeFixtures";

beforeEach(() => {
  useCopilotUIStore.setState({
    copilotLlmAuth: null,
    copilotLlmModel: "standard",
  });
});

afterEach(() => vi.restoreAllMocks());

describe("shared Advanced and ChatGPT Max offer", () => {
  it("presents both benefits with one billing action and never initiates a locked connection", async () => {
    const oauthRequest = vi.fn(() => HttpResponse.error());
    const openPopup = vi.spyOn(window, "open").mockReturnValue(null);
    mockMaxUpgrade([deploymentOffer(), lockedChatGPTOffer()]);
    server.use(http.get("*/integrations/codex/login", oauthRequest));
    render(<ConnectionPicker />);
    await openPicker();

    const chatGPT = await screen.findByText("Connect ChatGPT");
    expect(screen.getByText("Included with AutoGPT").isConnected).toBe(true);
    expect(screen.getAllByText("MAX")).toHaveLength(1);
    expect(
      screen.getByText("Use your existing ChatGPT plan.").isConnected,
    ).toBe(true);
    expect(screen.getByText("Save AutoGPT credits on chats.").isConnected).toBe(
      true,
    );
    const upgrades = screen.getAllByRole("link", { name: "Upgrade to Max" });
    expect(upgrades).toHaveLength(1);
    expect(upgrades[0].getAttribute("href")).toBe("/settings/billing");
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
    expect(screen.queryByText("Add a connection")).toBeNull();
    expect(screen.queryByText(lockedChatGPTOffer().lock_reason!)).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Connect a ChatGPT subscription" }),
    ).toBeNull();
    expect(screen.queryByRole("radio", { name: /ChatGPT/ })).toBeNull();
    expect(
      screen.queryByRole("radiogroup", {
        name: "Connection this chat runs on",
      }),
    ).toBeNull();

    const tiers = screen.getByRole("radiogroup", { name: "Model tier" });
    const advanced = within(tiers).getByRole("radio", {
      name: /Advanced.*opus-server/,
    });
    expect(advanced.getAttribute("aria-disabled")).toBe("true");
    expect(advanced.getAttribute("aria-checked")).toBe("false");
    expect(advanced.getAttribute("tabindex")).toBe("-1");
    expect(within(tiers).getAllByRole("radio")).toHaveLength(2);
    await userEvent.click(chatGPT);
    await userEvent.click(advanced);

    expect(oauthRequest).not.toHaveBeenCalled();
    expect(openPopup).not.toHaveBeenCalled();
    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard");
    expect(useCopilotUIStore.getState().copilotLlmAuth).toBeNull();
  });

  it("lets the keyboard reach and activate billing without selecting a locked tier", async () => {
    mockMaxUpgrade([deploymentOffer(), lockedChatGPTOffer()]);
    render(<ConnectionPicker />);
    await openPicker();

    const balanced = await screen.findByRole("radio", {
      name: "Balanced · sonnet-server",
    });
    const upgrade = screen.getByRole("link", { name: "Upgrade to Max" });
    expect(upgrade.closest('[role="radio"]')).toBeNull();
    balanced.focus();
    await userEvent.keyboard("{ArrowDown}{End}");
    expect(document.activeElement).toBe(balanced);
    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard");
    await userEvent.tab();
    expect(document.activeElement).toBe(upgrade);
    await userEvent.keyboard("{ArrowUp}{Home}");
    expect(document.activeElement).toBe(upgrade);
    const activateBilling = vi.fn((event: Event) => event.preventDefault());
    upgrade.addEventListener("click", activateBilling);
    await userEvent.keyboard("{Enter}");
    expect(activateBilling).toHaveBeenCalledOnce();
    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard");
  });

  it("offers only ChatGPT when Advanced is already available", async () => {
    mockMaxUpgrade([availableDeploymentOffer(), lockedChatGPTOffer()]);
    render(<ConnectionPicker />);
    await openPicker();

    expect((await screen.findByText("Connect ChatGPT")).isConnected).toBe(true);
    const advanced = screen.getByRole("radio", {
      name: "Advanced · opus-server",
    });
    await userEvent.click(advanced);
    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("advanced");
    expect(screen.getAllByText("Advanced")).toHaveLength(1);
    expect(
      screen.getAllByRole("link", { name: "Upgrade to Max" }),
    ).toHaveLength(1);
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Connect a ChatGPT subscription" }),
    ).toBeNull();
  });

  it("does not render an empty Max offer when identical deployment models hide the locked tier", async () => {
    const deployment = deploymentOffer();
    deployment.tiers = deployment.tiers.map((tier) => ({
      ...tier,
      display_model: "same-server-model",
    }));
    const linked = deploymentOffer({
      offer_id: "codex:cred-1",
      provider_family: "openai",
      display_name: "ChatGPT",
      auth_method: "chatgpt_oauth",
      credential_id: "cred-1",
      backed_by_label: "Your ChatGPT plan",
      is_default: false,
      tiers: availableDeploymentOffer().tiers,
    });
    mockMaxUpgrade([deployment, linked]);
    render(<ConnectionPicker />);
    await openPicker();

    expect(
      (await screen.findByRole("radio", { name: /ChatGPT/ })).isConnected,
    ).toBe(true);
    expect(screen.queryByRole("radiogroup", { name: "Model tier" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Upgrade to Max" })).toBeNull();
    expect(screen.queryByText("Included with AutoGPT")).toBeNull();
  });

  it.each([false, true])(
    "keeps a locked-only offer visible when connectionLocked is %s",
    async (connectionLocked) => {
      mockMaxUpgrade([lockedChatGPTOffer()]);
      render(<ConnectionPicker connectionLocked={connectionLocked} />);
      await openPicker();

      expect((await screen.findByText("Connect ChatGPT")).isConnected).toBe(
        true,
      );
      expect(
        screen
          .getByRole("link", { name: "Upgrade to Max" })
          .getAttribute("href"),
      ).toBe("/settings/billing");
      expect(
        screen.queryByRole("radiogroup", { name: "Model tier" }),
      ).toBeNull();
      expect(
        screen.queryByRole("button", {
          name: "Connect a ChatGPT subscription",
        }),
      ).toBeNull();
      expect(useCopilotUIStore.getState().copilotLlmAuth).toBeNull();
    },
  );
});

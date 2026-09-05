import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
  act,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../../store";
import {
  ComposerHarness,
  mockMaxUpgrade,
  proSubscription,
  openOffer,
  deploymentOffer,
  openPicker,
} from "./maxUpgradeFixtures";

beforeEach(() => {
  useCopilotUIStore.setState({
    copilotLlmAuth: null,
    copilotLlmModel: "standard",
  });
});

describe("contextual Max upgrade", () => {
  it("explains the locked server model and preserves the ChatGPT connection entry", async () => {
    mockMaxUpgrade();
    render(<ComposerHarness />);
    await openPicker();

    expect(
      await screen.findByRole("button", { name: "Upgrade to Max" }),
    ).toBeDefined();
    expect(screen.getByText("opus-server")).toBeDefined();
    expect(
      screen
        .getByRole("radio", { name: /Balanced.*sonnet-server/ })
        .getAttribute("aria-checked"),
    ).toBe("true");
    expect(screen.getByText("Add a connection")).toBeDefined();
    expect(screen.getByText("ChatGPT subscription")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Connect a ChatGPT subscription" }),
    ).toBeDefined();
  });

  it("does not invent a Max price when server pricing is unavailable", async () => {
    mockMaxUpgrade(proSubscription({ tier_costs: {} }));
    render(<ComposerHarness />);
    const offer = await openOffer();

    expect(offer.textContent).not.toMatch(/\$\d/);
    expect(
      within(offer)
        .getByRole("button", { name: "Review upgrade" })
        .hasAttribute("disabled"),
    ).toBe(true);
  });

  it("keeps the existing lock explanation for a non-Pro subscriber", async () => {
    mockMaxUpgrade(proSubscription({ tier: "BASIC" }));
    render(<ComposerHarness />);
    await openPicker();

    expect(
      await screen.findByText("A Max plan or higher is required for Advanced."),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Upgrade to Max" })).toBeNull();
    expect(screen.getByRole("link", { name: "See plans" })).toBeDefined();
  });

  it("keeps the server lock without guessed pricing while billing loads or fails", async () => {
    let respond: (() => void) | undefined;
    const response = new Promise<Response>((resolve) => {
      respond = () =>
        resolve(
          HttpResponse.json(
            { detail: "Subscription unavailable" },
            { status: 503 },
          ),
        );
    });
    const subscriptionRequest = vi.fn(() => response);
    mockMaxUpgrade();
    server.use(http.get("*/api/credits/subscription", subscriptionRequest));
    render(<ComposerHarness />);
    await openPicker();
    await waitFor(() => expect(subscriptionRequest).toHaveBeenCalled());

    expect(
      screen.getByText("A Max plan or higher is required for Advanced."),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Upgrade to Max" })).toBeNull();
    expect(screen.queryByText(/\$\d/)).toBeNull();

    await act(async () => {
      respond?.();
      await response;
    });
    expect(
      screen.getByText("A Max plan or higher is required for Advanced."),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Upgrade to Max" })).toBeNull();
    expect(screen.queryByText(/\$\d/)).toBeNull();
    expect(screen.getByRole("link", { name: "See plans" })).toBeDefined();
  });

  it("retains the ChatGPT plan gate alongside the contextual Advanced offer", async () => {
    mockMaxUpgrade(proSubscription(), [
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
    render(<ComposerHarness />);
    await openPicker();

    expect(
      await screen.findByRole("button", { name: "Upgrade to Max" }),
    ).toBeDefined();
    expect(
      screen.getByText("A Max plan or higher is required to use ChatGPT."),
    ).toBeDefined();
    expect(screen.getByRole("link", { name: "See plans" })).toBeDefined();
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
    mockMaxUpgrade(proSubscription(), [deploymentOffer(), linked]);
    render(<ComposerHarness />);
    await openPicker();
    await userEvent.click(
      await screen.findByRole("radio", { name: /ChatGPT/ }),
    );
    await userEvent.click(screen.getByRole("radio", { name: "Advanced" }));

    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("advanced");
    expect(screen.queryByRole("button", { name: "Upgrade to Max" })).toBeNull();
    expect(useCopilotUIStore.getState().copilotLlmAuth).toEqual({
      authProvider: "codex",
      credentialId: "cred-1",
    });
  });
});

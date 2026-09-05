import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../../store";
import {
  ComposerHarness,
  deploymentOffer,
  mockMaxUpgrade,
  proSubscription,
  openOffer,
  draft,
} from "./maxUpgradeFixtures";

beforeEach(() => {
  useCopilotUIStore.setState({
    copilotLlmAuth: null,
    copilotLlmModel: "standard",
  });
});

describe("Max upgrade payment", () => {
  it.each([
    {
      cycle: "monthly" as const,
      current: 5100,
      price: /\$247(?:\.00)?/,
      period: /month/,
    },
    {
      cycle: "yearly" as const,
      current: 52020,
      price: /\$2,519\.40/,
      period: /year/,
    },
  ])(
    "uses server $cycle pricing and charges only after explicit confirmation",
    async ({ cycle, current, price, period }) => {
      const upgrade = vi.fn();
      mockMaxUpgrade(
        proSubscription({ billing_cycle: cycle, monthly_cost: current }),
      );
      server.use(
        http.post("*/api/credits/subscription", async ({ request }) => {
          upgrade(await request.json());
          const subscription = proSubscription({
            tier: "MAX",
            billing_cycle: cycle,
          });
          const connection = deploymentOffer();
          connection.tiers = connection.tiers.map((tier) => ({
            ...tier,
            selectable: true,
            lock_reason: null,
          }));
          mockMaxUpgrade(subscription, [connection]);
          return HttpResponse.json(subscription);
        }),
      );
      render(<ComposerHarness />);
      await userEvent.type(
        screen.getByRole("textbox", { name: "Message" }),
        draft,
      );
      const offer = await openOffer();

      expect(offer.textContent).toMatch(price);
      expect(offer.textContent).toMatch(period);
      expect(within(offer).getByText("opus-server")).toBeDefined();
      expect(upgrade).not.toHaveBeenCalled();
      await userEvent.click(
        within(offer).getByRole("button", { name: "Review upgrade" }),
      );
      const confirmation = await screen.findByRole("dialog", {
        name: "Upgrade to Max?",
      });
      expect(confirmation.textContent).toMatch(/prorat/i);
      expect(confirmation.textContent).toMatch(/immediate|today/i);
      expect(upgrade).not.toHaveBeenCalled();

      await userEvent.click(
        within(confirmation).getByRole("button", { name: "Upgrade to Max" }),
      );
      await waitFor(() =>
        expect(upgrade).toHaveBeenCalledWith({
          tier: "MAX",
          billing_cycle: cycle,
        }),
      );
      expect(upgrade).toHaveBeenCalledTimes(1);
      expect(screen.getByLabelText<HTMLTextAreaElement>("Message").value).toBe(
        draft,
      );
      expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard");
      await waitFor(() =>
        expect(document.activeElement).toBe(
          screen.getByRole("button", { name: /— change/ }),
        ),
      );
      await userEvent.click(screen.getByRole("button", { name: /— change/ }));
      expect(
        screen
          .getByRole("radio", { name: "Advanced · opus-server" })
          .getAttribute("aria-checked"),
      ).toBe("false");
    },
  );

  it("cancels review without changing the plan, selected tier, or typed draft", async () => {
    const upgrade = vi.fn();
    mockMaxUpgrade();
    server.use(
      http.post("*/api/credits/subscription", () => {
        upgrade();
        return HttpResponse.json(proSubscription({ tier: "MAX" }));
      }),
    );
    render(<ComposerHarness />);
    await userEvent.type(
      screen.getByRole("textbox", { name: "Message" }),
      draft,
    );
    const offer = await openOffer();
    await userEvent.click(
      within(offer).getByRole("button", { name: "Review upgrade" }),
    );
    const confirmation = await screen.findByRole("dialog", {
      name: "Upgrade to Max?",
    });
    expect(confirmation.contains(document.activeElement)).toBe(true);
    await userEvent.click(
      within(confirmation).getByRole("button", { name: "Cancel" }),
    );
    const restoredOffer = await screen.findByRole("dialog", {
      name: "Unlock Advanced with Max.",
    });
    expect(restoredOffer.contains(document.activeElement)).toBe(true);
    await userEvent.click(
      within(restoredOffer).getByRole("button", { name: "Keep using Pro" }),
    );

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Unlock Advanced with Max." }),
      ).toBeNull(),
    );
    expect(upgrade).not.toHaveBeenCalled();
    expect(screen.getByLabelText<HTMLTextAreaElement>("Message").value).toBe(
      draft,
    );
    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard");
    expect(useCopilotUIStore.getState().copilotLlmAuth).toBeNull();
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Upgrade to Max" }),
      ),
    );
  });

  it("keeps the offer and an actionable error when payment fails", async () => {
    mockMaxUpgrade();
    server.use(
      http.post("*/api/credits/subscription", () =>
        HttpResponse.json(
          { detail: "Your card was declined." },
          { status: 402 },
        ),
      ),
    );
    render(<ComposerHarness />);
    await userEvent.type(
      screen.getByRole("textbox", { name: "Message" }),
      draft,
    );
    const offer = await openOffer();
    await userEvent.click(
      within(offer).getByRole("button", { name: "Review upgrade" }),
    );
    const confirmation = await screen.findByRole("dialog", {
      name: "Upgrade to Max?",
    });
    await userEvent.click(
      within(confirmation).getByRole("button", { name: "Upgrade to Max" }),
    );

    expect(await screen.findByText(/card was declined/i)).toBeDefined();
    const restoredOffer = await screen.findByRole("dialog", {
      name: "Unlock Advanced with Max.",
    });
    const billingLink = within(restoredOffer).getByRole("link", {
      name: "Open billing in a new tab",
    });
    expect(billingLink.getAttribute("href")).toBe("/settings/billing");
    expect(billingLink.getAttribute("target")).toBe("_blank");
    expect(screen.getByLabelText<HTMLTextAreaElement>("Message").value).toBe(
      draft,
    );
    expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard");
  });
});

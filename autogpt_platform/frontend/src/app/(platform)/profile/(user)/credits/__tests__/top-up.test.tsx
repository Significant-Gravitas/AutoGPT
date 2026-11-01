import { http, HttpResponse } from "msw";
import { describe, expect, test, vi } from "vitest";

import { server } from "@/mocks/mock-server";
import { restoreFromBackForwardCache } from "@/tests/integrations/back-forward-cache";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import CreditsPage from "../page";

const routerPush = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush, replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/profile/credits",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => true,
}));

describe("legacy credits page top-up", () => {
  test("keeps the top-up and plan buttons disabled while Stripe loads", async () => {
    const checkoutURL = "https://checkout.stripe.com/pay/cs_top_up";
    let topUps = 0;
    server.use(
      http.get("*/api/credits/subscription", () =>
        HttpResponse.json({
          tier: "BASIC",
          monthly_cost: 0,
          tier_costs: { BASIC: 0, PRO: 1999 },
          tier_multipliers: { BASIC: 1, PRO: 5 },
          proration_credit_cents: 0,
          pending_tier: null,
          pending_tier_effective_at: null,
        }),
      ),
      http.post("*/api/credits", () => {
        topUps += 1;
        return HttpResponse.json({ checkout_url: checkoutURL });
      }),
    );

    render(<CreditsPage />);
    const upgrade = await screen.findByRole("button", {
      name: /upgrade to pro/i,
    });
    const topUp = screen.getByRole("button", { name: "Top-up" });

    fireEvent.click(topUp);
    await waitFor(() => expect(routerPush).toHaveBeenCalledWith(checkoutURL));
    await act(async () => {});

    expect(topUp.hasAttribute("disabled")).toBe(true);
    expect(upgrade.hasAttribute("disabled")).toBe(true);
    fireEvent.submit(topUp.closest("form")!);
    await act(async () => {});
    expect(topUps).toBe(1);

    restoreFromBackForwardCache();

    expect(topUp.hasAttribute("disabled")).toBe(false);
    expect(upgrade.hasAttribute("disabled")).toBe(false);
  });

  test.each(["fails", "returns a checkout"])(
    "starts one checkout for a double click while the request is pending, and settles if it %s",
    async (outcome) => {
      const checkoutURL = "https://checkout.stripe.com/pay/cs_top_up";
      let topUps = 0;
      let finishTopUp = () => {};
      const pending = new Promise<void>((resolve) => {
        finishTopUp = resolve;
      });
      server.use(
        http.post("*/api/credits", async () => {
          topUps += 1;
          await pending;
          return outcome === "fails"
            ? HttpResponse.json({ detail: "Please try again" }, { status: 503 })
            : HttpResponse.json({ checkout_url: checkoutURL });
        }),
      );

      render(<CreditsPage />);
      const topUp = await screen.findByRole("button", { name: "Top-up" });
      const form = topUp.closest("form")!;

      fireEvent.submit(form);
      fireEvent.submit(form);
      await waitFor(() => expect(topUps).toBe(1));
      expect(topUp.hasAttribute("disabled")).toBe(true);
      fireEvent.submit(form);

      finishTopUp();
      if (outcome === "fails") {
        await waitFor(() => expect(topUp.hasAttribute("disabled")).toBe(false));
      } else {
        await waitFor(() =>
          expect(routerPush).toHaveBeenCalledWith(checkoutURL),
        );
        await act(async () => {});
        expect(topUp.hasAttribute("disabled")).toBe(true);
      }
      expect(topUps).toBe(1);
    },
  );
});

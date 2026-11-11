import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { LimitReachedExtras } from "../LimitReachedExtras";
import { PaidResetOffer } from "../PaidResetOffer";

// A local time far ahead, so it is dated the same in any time zone.
const WEEKLY_RESET = new Date(2030, 10, 12, 8, 30, 56);
const WITH_CREDITS = { price_credits: 8000, pay_with: "credits" as const };
const BY_CARD = { price_credits: 8000, pay_with: "card" as const };
const GRANTED = {
  status: "granted",
  paid_with: "credits",
  credits_charged: 8000,
  checkout_url: null,
  resets_at: WEEKLY_RESET.toISOString(),
  usage: null,
};
const CHECKOUT = {
  status: "payment_required",
  paid_with: "card",
  credits_charged: 0,
  checkout_url: "https://checkout.stripe.test/pay",
  resets_at: null,
  usage: null,
};

const prices: number[] = [];

function onPurchase(answer: (key: string) => Response | Promise<Response>) {
  const keys: string[] = [];
  prices.length = 0;
  server.use(
    http.post("*/api/chat/usage/paid-reset", async ({ request }) => {
      const body = (await request.json()) as {
        purchase_key: string;
        price_credits: number;
      };
      prices.push(body.price_credits);
      keys.push(body.purchase_key);
      return answer(body.purchase_key);
    }),
  );
  return keys;
}

function renderOffer(
  offer: typeof WITH_CREDITS | typeof BY_CARD = WITH_CREDITS,
) {
  const onBought = vi.fn();
  const goToCheckout = vi.fn();
  render(
    <PaidResetOffer
      offer={offer}
      weeklyResetsAt={WEEKLY_RESET}
      onBought={onBought}
      goToCheckout={goToCheckout}
    />,
  );
  return { onBought, goToCheckout };
}

async function openAndPay(label = "Pay $80 with credits") {
  fireEvent.click(
    screen.getByRole("button", { name: "Reset limits now · $80" }),
  );
  fireEvent.click(await screen.findByRole("button", { name: label }));
}

describe("buying a usage reset", () => {
  it("says what it gives, until when, and how it is paid before charging", async () => {
    const keys = onPurchase(() => HttpResponse.json(GRANTED));
    renderOffer();

    fireEvent.click(
      screen.getByRole("button", { name: "Reset limits now · $80" }),
    );

    expect(
      await screen.findByText(
        "Reset your limits now: a full week of usage, no daily limit, until your normal reset on Tue, Nov 12, 8:31 AM.",
      ),
    ).toBeDefined();
    expect(screen.getByText(/Your weekly reset doesn’t move\./)).toBeDefined();
    expect(
      screen.getByText("$80 is taken from your AutoGPT credit balance."),
    ).toBeDefined();
    expect(keys).toEqual([]);
  });

  it("pays from credits when they cover it and tells the caller", async () => {
    const keys = onPurchase(() => HttpResponse.json(GRANTED));
    const { onBought, goToCheckout } = renderOffer();

    await openAndPay();

    await waitFor(() => expect(onBought).toHaveBeenCalledTimes(1));
    expect(keys).toHaveLength(1);
    // The price on the button is the price agreed to.
    expect(prices).toEqual([8000]);
    expect(goToCheckout).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Pay $80 with credits" }),
      ).toBeNull(),
    );
  });

  it("says the card pays all of it when credits don’t cover it", async () => {
    onPurchase(() => HttpResponse.json(CHECKOUT));
    renderOffer(BY_CARD);

    fireEvent.click(
      screen.getByRole("button", { name: "Reset limits now · $80" }),
    );

    expect(
      await screen.findByText(
        "Your credit balance doesn’t cover $80, so you pay the full $80 by card on the next page.",
      ),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Pay $80 by card" }),
    ).toBeDefined();
  });

  it("goes to checkout for a card payment and resets nothing yet", async () => {
    const keys = onPurchase(() => HttpResponse.json(CHECKOUT));
    const { onBought, goToCheckout } = renderOffer(BY_CARD);

    await openAndPay("Pay $80 by card");

    await waitFor(() =>
      expect(goToCheckout).toHaveBeenCalledWith(
        "https://checkout.stripe.test/pay",
      ),
    );
    expect(onBought).not.toHaveBeenCalled();
    expect(keys).toHaveLength(1);
  });

  it("lets the customer pay again after coming back from checkout with Back", async () => {
    onPurchase(() => HttpResponse.json(CHECKOUT));
    const { goToCheckout } = renderOffer(BY_CARD);
    await openAndPay("Pay $80 by card");
    await waitFor(() => expect(goToCheckout).toHaveBeenCalledTimes(1));
    const pay = screen.getByRole("button", { name: /Pay \$80 by card/ });
    expect(pay.hasAttribute("disabled")).toBe(true);

    // The browser restores the page from its back/forward cache.
    window.dispatchEvent(
      Object.assign(new Event("pageshow"), { persisted: true }),
    );

    await waitFor(() =>
      expect(
        screen
          .getByRole("button", { name: /Pay \$80 by card/ })
          .hasAttribute("disabled"),
      ).toBe(false),
    );
  });

  it("goes to checkout when the balance was spent since the offer was shown", async () => {
    onPurchase(() => HttpResponse.json(CHECKOUT));
    const { onBought, goToCheckout } = renderOffer(WITH_CREDITS);

    await openAndPay();

    await waitFor(() => expect(goToCheckout).toHaveBeenCalledTimes(1));
    expect(onBought).not.toHaveBeenCalled();
  });

  it("treats a card answer without a checkout as a failure", async () => {
    onPurchase(() => HttpResponse.json({ ...CHECKOUT, checkout_url: null }));
    const { onBought, goToCheckout } = renderOffer(BY_CARD);

    await openAndPay("Pay $80 by card");

    expect(
      await screen.findByRole("button", { name: "Pay $80 by card" }),
    ).toBeDefined();
    expect(goToCheckout).not.toHaveBeenCalled();
    expect(onBought).not.toHaveBeenCalled();
  });

  it("closes without buying when the price is no longer the one shown", async () => {
    const keys = onPurchase(() =>
      HttpResponse.json({ detail: "price_changed" }, { status: 409 }),
    );
    const { onBought, goToCheckout } = renderOffer();

    await openAndPay();

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Pay $80 with credits" }),
      ).toBeNull(),
    );
    expect(keys).toHaveLength(1);
    expect(onBought).not.toHaveBeenCalled();
    expect(goToCheckout).not.toHaveBeenCalled();
  });

  it("repeats the same purchase after a failure, so a retry can't buy twice", async () => {
    let attempts = 0;
    const keys = onPurchase(() => {
      attempts += 1;
      return attempts === 1
        ? HttpResponse.json({ detail: "boom" }, { status: 500 })
        : HttpResponse.json(GRANTED);
    });
    const { onBought } = renderOffer();

    await openAndPay();
    await waitFor(() => expect(keys).toHaveLength(1));
    expect(onBought).not.toHaveBeenCalled();
    fireEvent.click(
      await screen.findByRole("button", { name: "Pay $80 with credits" }),
    );

    await waitFor(() => expect(onBought).toHaveBeenCalledTimes(1));
    expect(keys).toHaveLength(2);
    expect(keys[1]).toBe(keys[0]);
  });

  it("uses a new key for the next, separate purchase", async () => {
    const keys = onPurchase(() => HttpResponse.json(GRANTED));
    const { onBought } = renderOffer();

    await openAndPay();
    await waitFor(() => expect(onBought).toHaveBeenCalledTimes(1));
    await openAndPay();
    await waitFor(() => expect(onBought).toHaveBeenCalledTimes(2));

    expect(keys[1]).not.toBe(keys[0]);
  });
});

describe("what a reached limit offers", () => {
  const usage = {
    weekly: { resets_at: WEEKLY_RESET },
    reserve: { resets_at: new Date(2030, 10, 5, 20, 30, 56) },
    reset_offer: { price_credits: 1250, pay_with: "card" as const },
  };

  it("offers a reset and says when the reserve refills on a weekly limit", () => {
    render(<LimitReachedExtras window="weekly" usage={usage} />);

    expect(
      screen.getByText(
        "Need to keep going? Reset your daily and weekly limits now.",
      ),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Reset limits now · $12.50" }),
    ).toBeDefined();
    expect(
      screen.getByText("A small daily reserve refills Tue, Nov 5, 8:31 PM."),
    ).toBeDefined();
  });

  it("offers a reset on a daily limit too, without the reserve line", () => {
    render(<LimitReachedExtras window="daily" usage={usage} />);

    expect(
      screen.getByRole("button", { name: "Reset limits now · $12.50" }),
    ).toBeDefined();
    expect(screen.queryByText(/daily reserve refills/)).toBeNull();
  });

  it("shows nothing for a plan with neither a reserve nor an offer", () => {
    const { container } = render(
      <LimitReachedExtras
        window="weekly"
        usage={{ reserve: null, reset_offer: null }}
      />,
    );

    expect(container.textContent).toBe("");
  });
});

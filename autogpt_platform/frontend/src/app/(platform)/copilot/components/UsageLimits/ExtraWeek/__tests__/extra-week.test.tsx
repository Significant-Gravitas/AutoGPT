import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { TopUpPromptContext } from "@/components/layout/TopUpPrompt/useTopUpPrompt";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { ExtraWeekOffer } from "../ExtraWeekOffer";
import { WeeklyLimitExtras } from "../WeeklyLimitExtras";

const MAX_OFFER = { price_credits: 8000 };
const BOUGHT = {
  credits_charged: 8000,
  remaining_balance: 2000,
  expires_at: "2026-12-04T20:30:56Z",
  usage: { tier: "MAX", daily: null, weekly: null, state: "ok" },
};

const prices: number[] = [];

function onPurchase(answer: (key: string) => Response | Promise<Response>) {
  const keys: string[] = [];
  prices.length = 0;
  server.use(
    http.post("*/api/chat/usage/extra-week", async ({ request }) => {
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

function renderOffer(onBought = vi.fn()) {
  const openTopUp = vi.fn();
  render(
    <TopUpPromptContext.Provider
      value={{ isOutOfCredits: false, openTopUp, closeTopUp: vi.fn() }}
    >
      <ExtraWeekOffer offer={MAX_OFFER} onBought={onBought} />
    </TopUpPromptContext.Provider>,
  );
  return { openTopUp, onBought };
}

async function openAndPay() {
  fireEvent.click(
    screen.getByRole("button", { name: "Add an extra week · $80" }),
  );
  fireEvent.click(await screen.findByRole("button", { name: "Pay $80" }));
}

describe("buying an extra week", () => {
  it("states the price and the terms before anything is charged", async () => {
    const keys = onPurchase(() => HttpResponse.json(BOUGHT));
    renderOffer();

    fireEvent.click(
      screen.getByRole("button", { name: "Add an extra week · $80" }),
    );

    expect(
      await screen.findByText(/\$80 from your AutoGPT credit balance/),
    ).toBeDefined();
    expect(
      screen.getByText("Used after your weekly limit runs out."),
    ).toBeDefined();
    expect(
      screen.getByText("Your daily limit doesn’t apply to it."),
    ).toBeDefined();
    expect(
      screen.getByText("Lasts 30 days from today, through your usage resets."),
    ).toBeDefined();
    expect(keys).toEqual([]);
  });

  it("buys it once confirmed and tells the caller", async () => {
    const keys = onPurchase(() => HttpResponse.json(BOUGHT));
    const { onBought, openTopUp } = renderOffer();

    await openAndPay();

    await waitFor(() => expect(onBought).toHaveBeenCalledTimes(1));
    expect(keys).toHaveLength(1);
    // The price on the button is the price agreed to.
    expect(prices).toEqual([8000]);
    expect(openTopUp).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Pay $80" })).toBeNull(),
    );
  });

  it("opens the top-up when credits are short, and charges nothing", async () => {
    const keys = onPurchase(() =>
      HttpResponse.json(
        { detail: { message: "Not enough", needed: 8000, balance: 300 } },
        { status: 402 },
      ),
    );
    const { onBought, openTopUp } = renderOffer();

    await openAndPay();

    await waitFor(() => expect(openTopUp).toHaveBeenCalledTimes(1));
    expect(onBought).not.toHaveBeenCalled();
    expect(keys).toHaveLength(1);
  });

  it("closes without buying when the price is no longer the one shown", async () => {
    const keys = onPurchase(() =>
      HttpResponse.json({ detail: "price_changed" }, { status: 409 }),
    );
    const { onBought, openTopUp } = renderOffer();

    await openAndPay();

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Pay $80" })).toBeNull(),
    );
    expect(keys).toHaveLength(1);
    expect(onBought).not.toHaveBeenCalled();
    expect(openTopUp).not.toHaveBeenCalled();
  });

  it("repeats the same purchase after a failure, so a retry can't buy twice", async () => {
    let attempts = 0;
    const keys = onPurchase(() => {
      attempts += 1;
      return attempts === 1
        ? HttpResponse.json({ detail: "boom" }, { status: 500 })
        : HttpResponse.json(BOUGHT);
    });
    const { onBought } = renderOffer();

    await openAndPay();
    await waitFor(() => expect(keys).toHaveLength(1));
    expect(onBought).not.toHaveBeenCalled();
    fireEvent.click(await screen.findByRole("button", { name: "Pay $80" }));

    await waitFor(() => expect(onBought).toHaveBeenCalledTimes(1));
    expect(keys).toHaveLength(2);
    expect(keys[1]).toBe(keys[0]);
  });

  it("uses a new key for the next, separate purchase", async () => {
    const keys = onPurchase(() => HttpResponse.json(BOUGHT));
    const { onBought } = renderOffer();

    await openAndPay();
    await waitFor(() => expect(onBought).toHaveBeenCalledTimes(1));
    await openAndPay();
    await waitFor(() => expect(onBought).toHaveBeenCalledTimes(2));

    expect(keys[1]).not.toBe(keys[0]);
  });
});

describe("what a used-up limit offers", () => {
  const usage = {
    reserve: { resets_at: new Date(2030, 10, 5, 20, 30, 56) },
    extra_week_offer: { price_credits: 1250 },
  };

  it("offers another week and says when the reserve refills on a weekly limit", () => {
    render(<WeeklyLimitExtras window="weekly" usage={usage} />);

    expect(
      screen.getByText("Out of usage? Add another week now."),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Add an extra week · $12.50" }),
    ).toBeDefined();
    expect(
      screen.getByText("A small daily reserve refills Tue, Nov 5, 8:31 PM."),
    ).toBeDefined();
  });

  it("offers nothing on a daily limit, which an extra week wouldn't lift", () => {
    const { container } = render(
      <WeeklyLimitExtras window="daily" usage={usage} />,
    );

    expect(container.textContent).toBe("");
  });

  it("shows nothing for a plan with neither a reserve nor an offer", () => {
    const { container } = render(
      <WeeklyLimitExtras
        window="weekly"
        usage={{ reserve: null, extra_week_offer: null }}
      />,
    );

    expect(container.textContent).toBe("");
  });
});

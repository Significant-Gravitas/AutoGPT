import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  act,
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { restoreFromBackForwardCache } from "@/tests/integrations/back-forward-cache";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";

import { TopUpForm } from "../TopUpForm";

const routerPush = vi.hoisted(() => vi.fn());

const CHECKOUT_URL = "https://checkout.stripe.com/pay/cs_test_top_up";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush, replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

function captureTopUpRequest() {
  let body: unknown = null;
  server.use(
    http.post("*/api/credits", async ({ request }) => {
      body = await request.json();
      return HttpResponse.json({ checkout_url: CHECKOUT_URL });
    }),
  );
  return { getBody: () => body };
}

beforeEach(() => {
  routerPush.mockClear();
  useAuthStore.setState({ user: { id: "user-a" } as User });
});

afterEach(() => {
  useAuthStore.setState({ user: null });
});

describe("TopUpForm", () => {
  test("submits the entered amount converted to cents", async () => {
    const { getBody } = captureTopUpRequest();

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "15" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));

    await waitFor(() => expect(getBody()).toEqual({ credit_amount: 1500 }));
    await waitFor(() => expect(routerPush).toHaveBeenCalledWith(CHECKOUT_URL));
    await act(async () => {});
  });

  test("rejects amounts below the $5 minimum without calling the API", async () => {
    const { getBody } = captureTopUpRequest();

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "3" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));

    expect(await screen.findByText(/Top-ups start at \$5/i)).toBeDefined();
    expect(getBody()).toBeNull();
  });

  test("disables the submit button while the checkout request is in flight", async () => {
    // Hold the request open so the in-flight loading state is observable.
    server.use(
      http.post("*/api/credits", async () => {
        await new Promise((resolve) => setTimeout(resolve, 300));
        return HttpResponse.json({ checkout_url: CHECKOUT_URL });
      }),
    );

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "15" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));

    const button = await screen.findByRole("button", { name: /redirecting/i });
    expect((button as HTMLButtonElement).disabled).toBe(true);

    // Let the held request finish inside the test: settling it after the
    // file's environment is torn down fails the whole run.
    await waitFor(() => expect(routerPush).toHaveBeenCalledWith(CHECKOUT_URL), {
      timeout: 2000,
    });
    await act(async () => {});
  });

  test("stays busy while Stripe loads, and is usable again after Back restores the page", async () => {
    const checkoutURL = "https://checkout.stripe.com/pay/cs_top_up";
    let topUps = 0;
    server.use(
      http.post("*/api/credits", () => {
        topUps += 1;
        return HttpResponse.json({ checkout_url: checkoutURL });
      }),
    );

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "15" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));
    await waitFor(() => expect(routerPush).toHaveBeenCalledWith(checkoutURL));
    await act(async () => {});

    const busy = screen.getByRole("button", { name: /redirecting/i });
    expect(busy.hasAttribute("disabled")).toBe(true);
    fireEvent.click(busy);
    expect(topUps).toBe(1);

    restoreFromBackForwardCache();

    const topUp = screen.getByRole("button", { name: /top up/i });
    expect(topUp.hasAttribute("disabled")).toBe(false);
    fireEvent.click(topUp);
    await waitFor(() => expect(topUps).toBe(2));
  });

  test("releases the button without leaving when no checkout URL comes back", async () => {
    let topUps = 0;
    server.use(
      http.post("*/api/credits", () => {
        topUps += 1;
        return HttpResponse.json({ checkout_url: "" });
      }),
    );

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "15" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));
    await waitFor(() => expect(topUps).toBe(1));
    await act(async () => {});

    const topUp = await screen.findByRole("button", { name: /top up/i });
    expect(topUp.hasAttribute("disabled")).toBe(false);
    expect(routerPush).not.toHaveBeenCalledWith("");
  });

  test("ignores a checkout started by the previous identity and leaves the page usable", async () => {
    const oldCheckoutURL = "https://checkout.example/old-user";
    const newCheckoutURL = "https://checkout.example/new-user";
    let topUps = 0;
    let releaseOldRequest: (() => void) | undefined;
    server.use(
      http.post("*/api/credits", async () => {
        topUps += 1;
        if (topUps > 1) {
          return HttpResponse.json({ checkout_url: newCheckoutURL });
        }
        await new Promise<void>((resolve) => {
          releaseOldRequest = resolve;
        });
        return HttpResponse.json({ checkout_url: oldCheckoutURL });
      }),
    );
    const requestTopUp = vi.spyOn(AutoGPTServerAPI.prototype, "requestTopUp");

    try {
      render(<TopUpForm />);
      fireEvent.change(screen.getByLabelText("Amount"), {
        target: { value: "15" },
      });
      fireEvent.click(screen.getByRole("button", { name: /top up/i }));
      await screen.findByRole("button", { name: /redirecting/i });
      await waitFor(() => expect(releaseOldRequest).toBeDefined());
      const oldCheckout = requestTopUp.mock.results[0]?.value;

      act(() => {
        useAuthStore.setState({ user: { id: "user-b" } as User });
      });
      releaseOldRequest?.();
      await act(async () => {
        await oldCheckout;
      });
      await act(async () => {});

      expect(routerPush).not.toHaveBeenCalled();
      // A stale answer must not mark the page as leaving for Stripe, or every
      // checkout button would stay busy for the next account.
      const topUp = screen.getByRole("button", { name: /top up/i });
      expect(topUp.hasAttribute("disabled")).toBe(false);

      fireEvent.change(screen.getByLabelText("Amount"), {
        target: { value: "20" },
      });
      fireEvent.click(topUp);
      await waitFor(() =>
        expect(routerPush).toHaveBeenCalledWith(newCheckoutURL),
      );
      expect(routerPush).not.toHaveBeenCalledWith(oldCheckoutURL);
      await act(async () => {});
    } finally {
      requestTopUp.mockRestore();
    }
  });
});

import { http, HttpResponse } from "msw";
import { describe, expect, test, vi } from "vitest";
import {
  act,
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getPostV1RequestCreditTopUpMockHandler } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { restoreFromBackForwardCache } from "@/tests/integrations/back-forward-cache";

import { TopUpForm } from "../TopUpForm";

const routerPush = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush, replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

function captureTopUpRequest() {
  let body: unknown = null;
  server.use(
    getPostV1RequestCreditTopUpMockHandler(
      async (info: { request: Request }) => {
        body = await info.request.json();
      },
    ),
  );
  return { getBody: () => body };
}

describe("TopUpForm", () => {
  test("submits the entered amount converted to cents", async () => {
    const { getBody } = captureTopUpRequest();

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "15" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));

    await waitFor(() => expect(getBody()).toEqual({ credit_amount: 1500 }));
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
      getPostV1RequestCreditTopUpMockHandler(
        () => new Promise((resolve) => setTimeout(resolve, 300)),
      ),
    );

    render(<TopUpForm />);

    fireEvent.change(screen.getByLabelText("Amount"), {
      target: { value: "15" },
    });
    fireEvent.click(screen.getByRole("button", { name: /top up/i }));

    const button = await screen.findByRole("button", { name: /redirecting/i });
    expect((button as HTMLButtonElement).disabled).toBe(true);
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
});

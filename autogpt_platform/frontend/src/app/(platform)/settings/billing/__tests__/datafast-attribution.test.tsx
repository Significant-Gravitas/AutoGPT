/**
 * Page-level integration test for DataFast attribution on the credit top-up.
 *
 * The credit top-up flows through `usePostV1RequestCreditTopUp` →
 * `customMutator`, which attaches the DataFast headers returned by
 * `getDatafastAttribution()` (read from the `datafast_visitor_id` /
 * `datafast_session_id` cookies). MSW intercepts the outgoing POST so we can
 * assert which headers were attached.
 *
 * We drive the real BalanceCard UI (Add credits → amount input → Continue to
 * checkout) rather than calling the hook directly, and stub `document.cookie`
 * the same way the datafast-attribution unit test does.
 *
 * Redirect guard: the POST handler returns `{ checkout_url: null }` so the
 * hook treats it as a non-redirect error and never sets
 * `window.location.href` — the request (and its headers) is still captured.
 */

import { fireEvent } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";

import { BalanceCard } from "../components/AutomationCreditsTab/BalanceCard/BalanceCard";

function stubCookies(value: string) {
  vi.spyOn(document, "cookie", "get").mockReturnValue(value);
}

async function submitTopUp() {
  fireEvent.click(await screen.findByRole("button", { name: /add credits/i }));

  const amountInput = screen.getByPlaceholderText(/amount/i);
  fireEvent.change(amountInput, { target: { value: "20" } });

  const continueButton = await screen.findByRole("button", {
    name: /continue to checkout/i,
  });
  await waitFor(() =>
    expect(continueButton.hasAttribute("disabled")).toBe(false),
  );
  fireEvent.click(continueButton);
}

afterEach(() => vi.restoreAllMocks());

describe("DataFast attribution on credit top-up", () => {
  it("attaches the DataFast headers when the attribution cookies are present", async () => {
    stubCookies("datafast_visitor_id=vis_1; datafast_session_id=ses_1");

    const captured: { headers: Headers | null } = { headers: null };
    server.use(
      http.get("*/api/credits", () => HttpResponse.json({ credits: 1000 })),
      http.post("*/api/credits", async ({ request }) => {
        captured.headers = request.headers;
        return HttpResponse.json({ checkout_url: null });
      }),
    );

    render(<BalanceCard />);
    await submitTopUp();

    await waitFor(() => expect(captured.headers).not.toBeNull());
    expect(captured.headers?.get("X-Datafast-Visitor-Id")).toBe("vis_1");
    expect(captured.headers?.get("X-Datafast-Session-Id")).toBe("ses_1");
  });

  it("omits the DataFast headers when no attribution cookies exist", async () => {
    stubCookies("");

    const captured: { headers: Headers | null } = { headers: null };
    server.use(
      http.get("*/api/credits", () => HttpResponse.json({ credits: 1000 })),
      http.post("*/api/credits", async ({ request }) => {
        captured.headers = request.headers;
        return HttpResponse.json({ checkout_url: null });
      }),
    );

    render(<BalanceCard />);
    await submitTopUp();

    await waitFor(() => expect(captured.headers).not.toBeNull());
    expect(captured.headers?.get("X-Datafast-Visitor-Id")).toBeNull();
    expect(captured.headers?.get("X-Datafast-Session-Id")).toBeNull();
  });
});

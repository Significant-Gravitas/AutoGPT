import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "@/mocks/mock-server";
import { ProActivationProvider } from "../ProActivationProvider";
import { useProActivation } from "../useProActivation";
import { quote } from "./fixtures";

vi.mock("@/lib/auth/hooks/useAuthStore", () => ({
  useAuthStore: Object.assign(
    (select: (state: { user: { id: string } }) => unknown) =>
      select({ user: { id: "user-1" } }),
    { getState: () => ({ user: { id: "user-1" } }) },
  ),
}));
vi.mock("@/components/molecules/Dialog/Dialog", () => ({
  Dialog: Object.assign(
    function Dialog({
      controlled,
      children,
      title,
    }: {
      controlled: { isOpen: boolean };
      children: React.ReactNode;
      title: React.ReactNode;
    }) {
      return controlled.isOpen ? (
        <div role="dialog">
          {title}
          {children}
        </div>
      ) : null;
    },
    {
      Content: ({ children }: { children: React.ReactNode }) => (
        <div>{children}</div>
      ),
    },
  ),
}));

function Subject() {
  const activation = useProActivation();
  return (
    <>
      <button
        onClick={() => activation.start("/copilot/thread?resume=1#draft")}
      >
        Upgrade
      </button>
      <span>
        {activation.isReady ? "allowance ready" : "allowance waiting"}
      </span>
    </>
  );
}
function mount() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ProActivationProvider>
        <Subject />
      </ProActivationProvider>
    </QueryClientProvider>,
  );
}
const base = "*/api/proxy/api/credits/pro-activation";
const confirm = vi.fn();
const preview = vi.fn();

beforeEach(() => {
  sessionStorage.clear();
  window.history.replaceState({}, "", "/");
  confirm.mockReset();
  preview.mockReset();
  server.use(
    http.get(`${base}/current`, () => new HttpResponse(null, { status: 404 })),
    http.post(`${base}/preview`, async ({ request }) => {
      preview(await request.json());
      return HttpResponse.json(quote);
    }),
    http.post(`${base}/attempt-1/confirm`, async ({ request }) => {
      confirm(await request.json());
      return HttpResponse.json({
        ...quote,
        status: "processing",
        retry_after_seconds: 30,
      });
    }),
    http.get(`${base}/attempt-1`, () =>
      HttpResponse.json({
        ...quote,
        status: "processing",
        retry_after_seconds: 30,
      }),
    ),
  );
});

describe("Pro activation", () => {
  it.each(["processing", "payment_required", "action_required", "failed"])(
    "ignores unsolicited recovered %s outside an activation flow",
    async (status) => {
      const current = vi.fn();
      server.use(
        http.get(`${base}/current`, () => {
          current();
          return HttpResponse.json({ ...quote, status });
        }),
      );
      mount();
      await waitFor(() => expect(current).toHaveBeenCalledTimes(1));
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(screen.getByText("allowance waiting")).toBeDefined();
      expect(confirm).not.toHaveBeenCalled();
      expect(preview).not.toHaveBeenCalled();
    },
  );

  it("retains the checkout-return signal while the billing page cleans up its URL", async () => {
    window.history.replaceState(
      {},
      "",
      "/settings/billing?subscription=success",
    );
    let releaseCurrent: () => void = () => {};
    const pending = new Promise<void>((resolve) => {
      releaseCurrent = resolve;
    });
    const currentStarted = vi.fn();
    server.use(
      http.get(`${base}/current`, async () => {
        currentStarted();
        await pending;
        return HttpResponse.json({ ...quote, status: "ready" });
      }),
      http.get("*/api/proxy/api/credits/trial", () =>
        HttpResponse.json({ active: false, converted: true }),
      ),
      http.get("*/api/proxy/api/credits/subscription", () =>
        HttpResponse.json({ tier: "PRO" }),
      ),
      http.get("*/api/proxy/api/chat/usage", () =>
        HttpResponse.json({ tier: "PRO" }),
      ),
    );
    mount();
    await waitFor(() => expect(currentStarted).toHaveBeenCalledTimes(1));
    window.history.replaceState({}, "", "/settings/billing");
    releaseCurrent();
    await screen.findByText("allowance ready");
    expect(screen.getByText("You’re ready to keep going.")).toBeDefined();
    fireEvent.click(
      screen.getByRole("button", { name: "Continue where you left off" }),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    // Closing success must not immediately replace fresh-Pro billing with a Max upsell.
    expect(screen.getByText("allowance ready")).toBeDefined();
  });

  it("does not announce readiness when refreshed data still describes the trial", async () => {
    server.use(
      http.get(`${base}/current`, () =>
        HttpResponse.json({ ...quote, status: "ready" }),
      ),
      http.get("*/api/proxy/api/credits/trial", () =>
        HttpResponse.json({ active: true, converted: false }),
      ),
      http.get("*/api/proxy/api/credits/subscription", () =>
        HttpResponse.json({ tier: "TRIAL" }),
      ),
      http.get("*/api/proxy/api/chat/usage", () =>
        HttpResponse.json({ tier: "TRIAL" }),
      ),
    );
    sessionStorage.setItem(
      "pro-activation:user-1",
      JSON.stringify({ id: quote.id, token: quote.terms_token }),
    );
    mount();
    await screen.findByText(
      /Your payment is confirmed. We’re refreshing your plan/,
    );
    expect(screen.queryByText("allowance ready")).toBeNull();
    expect(screen.getByText("allowance waiting")).toBeDefined();
  });

  it.each(["PRO", "MAX", "NO_TIER"])(
    "ignores a historical completed activation for current %s customers",
    async (tier) => {
      const readUsage = vi.fn();
      const current = vi.fn();
      server.use(
        http.get(`${base}/current`, () => {
          current();
          return HttpResponse.json({ ...quote, status: "ready" });
        }),
        http.get("*/api/proxy/api/credits/trial", () =>
          HttpResponse.json({ active: false, converted: true }),
        ),
        http.get("*/api/proxy/api/credits/subscription", () =>
          HttpResponse.json({ tier }),
        ),
        http.get("*/api/proxy/api/chat/usage", () => {
          readUsage();
          return HttpResponse.json({ tier });
        }),
      );
      mount();
      await waitFor(() => expect(current).toHaveBeenCalledTimes(1));
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(readUsage).not.toHaveBeenCalled();
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(screen.queryByText("allowance ready")).toBeNull();
    },
  );

  it("ignores a stale mount recovery response after an explicit newer upgrade", async () => {
    let releaseOld: () => void = () => {};
    const pending = new Promise<void>((resolve) => {
      releaseOld = resolve;
    });
    let calls = 0;
    server.use(
      http.get(`${base}/current`, async () => {
        calls++;
        if (calls === 1) {
          await pending;
          return HttpResponse.json({ ...quote, status: "failed" });
        }
        return new HttpResponse(null, { status: 404 });
      }),
    );
    mount();
    await waitFor(() => expect(calls).toBe(1));
    fireEvent.click(screen.getByText("Upgrade"));
    await screen.findByRole("button", { name: "Pay £61.23 & start Pro" });
    releaseOld();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(
      screen.getByRole("button", { name: "Pay £61.23 & start Pro" }),
    ).toBeDefined();
    expect(screen.queryByText("Your payment wasn’t completed.")).toBeNull();
  });

  it("retains explicit consent across refresh and retries the same attempt and token", async () => {
    sessionStorage.setItem(
      "pro-activation:user-1",
      JSON.stringify({ id: quote.id, token: quote.terms_token }),
    );
    server.use(http.get(`${base}/current`, () => HttpResponse.json(quote)));
    mount();
    fireEvent.click(
      await screen.findByRole("button", { name: "Retry same confirmation" }),
    );
    await screen.findByRole("button", { name: "Check activation status" });
    expect(confirm).toHaveBeenCalledExactlyOnceWith({
      confirmed: true,
      terms_token: quote.terms_token,
    });
    expect(preview).not.toHaveBeenCalled();
  });

  it("requires refreshed terms after a conflict and never automatically retries confirmation", async () => {
    server.use(
      http.post(`${base}/attempt-1/confirm`, async ({ request }) => {
        confirm(await request.json());
        return HttpResponse.json({ detail: "Terms expired" }, { status: 409 });
      }),
      http.get(`${base}/attempt-1`, () => HttpResponse.json(quote)),
    );
    mount();
    fireEvent.click(screen.getByText("Upgrade"));
    fireEvent.click(
      await screen.findByRole("button", { name: "Pay £61.23 & start Pro" }),
    );
    const refresh = await screen.findByRole("button", {
      name: "Review current terms",
    });
    expect(
      screen
        .getByRole("button", { name: "Pay £61.23 & start Pro" })
        .hasAttribute("disabled"),
    ).toBe(true);
    expect(confirm).toHaveBeenCalledTimes(1);
    fireEvent.click(refresh);
    await waitFor(() => expect(preview).toHaveBeenCalledTimes(2));
    expect(confirm).toHaveBeenCalledTimes(1);
  });

  it("shows the actual quote and modifiers before explicit consent", async () => {
    mount();
    fireEvent.click(screen.getByText("Upgrade"));
    const pay = await screen.findByRole("button", {
      name: "Pay £61.23 & start Pro",
    });
    expect(screen.getByText("£85.00 / year")).toBeDefined();
    expect(screen.getByText(/20% off the first invoice only/)).toBeDefined();
    expect(screen.getByText(/VAT: 20% additional · GB/)).toBeDefined();
    expect(confirm).not.toHaveBeenCalled();
    expect(preview).toHaveBeenCalledWith({
      return_to: "/copilot/thread?resume=1#draft",
    });
    fireEvent.click(pay);
    await screen.findByRole("button", { name: "Check activation status" });
    expect(confirm).toHaveBeenCalledExactlyOnceWith({
      confirmed: true,
      terms_token: quote.terms_token,
    });
    expect(screen.getByText("allowance waiting")).toBeDefined();
  });

  it("waits for all authoritative data before declaring readiness", async () => {
    let releaseUsage: () => void = () => {};
    const usageWait = new Promise<void>((resolve) => {
      releaseUsage = resolve;
    });
    server.use(
      http.get(`${base}/attempt-1`, () =>
        HttpResponse.json({ ...quote, status: "ready" }),
      ),
      http.get("*/api/proxy/api/credits/trial", () =>
        HttpResponse.json({ status: "converted" }),
      ),
      http.get("*/api/proxy/api/credits/subscription", () =>
        HttpResponse.json({ tier: "PRO" }),
      ),
      http.get("*/api/proxy/api/chat/usage", async () => {
        await usageWait;
        return HttpResponse.json({ tier: "PRO", daily: { used: 0 } });
      }),
    );
    mount();
    fireEvent.click(screen.getByText("Upgrade"));
    fireEvent.click(
      await screen.findByRole("button", { name: "Pay £61.23 & start Pro" }),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Check activation status" }),
    );
    expect(screen.getByText("allowance waiting")).toBeDefined();
    expect(screen.queryByText("You’re ready to keep going.")).toBeNull();
    releaseUsage();
    await screen.findByText("allowance ready");
    expect(
      await screen.findByText("You’re ready to keep going."),
    ).toBeDefined();
  });

  it("recovers existing authentication using its invoice without another charge", async () => {
    sessionStorage.setItem(
      "pro-activation:user-1",
      JSON.stringify({ id: quote.id, token: quote.terms_token }),
    );
    server.use(
      http.get(`${base}/current`, () =>
        HttpResponse.json({
          ...quote,
          status: "action_required",
          hosted_invoice_url: "https://invoice.stripe.com/i/existing",
        }),
      ),
    );
    mount();
    const link = await screen.findByRole("link", {
      name: "Verify payment securely",
    });
    expect(link.getAttribute("href")).toBe(
      "https://invoice.stripe.com/i/existing",
    );
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noopener noreferrer");
    expect(screen.getByText(/keeping your draft here/)).toBeDefined();
    expect(confirm).not.toHaveBeenCalled();
    expect(preview).not.toHaveBeenCalled();
    expect(screen.getByText("allowance waiting")).toBeDefined();
  });

  it("recovers processing after navigation using GET only", async () => {
    sessionStorage.setItem(
      "pro-activation:user-1",
      JSON.stringify({ id: quote.id, token: quote.terms_token }),
    );
    server.use(
      http.get(`${base}/current`, () =>
        HttpResponse.json({
          ...quote,
          status: "processing",
          retry_after_seconds: 30,
        }),
      ),
    );
    mount();
    fireEvent.click(
      await screen.findByRole("button", { name: "Check activation status" }),
    );
    await waitFor(() => expect(confirm).not.toHaveBeenCalled());
    expect(preview).not.toHaveBeenCalled();
    expect(screen.getByText("allowance waiting")).toBeDefined();
  });

  it("resolves a lost confirmation response using the same attempt's status", async () => {
    server.use(
      http.post(`${base}/attempt-1/confirm`, async ({ request }) => {
        confirm(await request.json());
        return HttpResponse.error();
      }),
    );
    mount();
    fireEvent.click(screen.getByText("Upgrade"));
    fireEvent.click(
      await screen.findByRole("button", { name: "Pay £61.23 & start Pro" }),
    );
    await screen.findByRole("button", { name: "Check activation status" });
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(preview).toHaveBeenCalledTimes(1);
    expect(screen.getByText("allowance waiting")).toBeDefined();
  });
});

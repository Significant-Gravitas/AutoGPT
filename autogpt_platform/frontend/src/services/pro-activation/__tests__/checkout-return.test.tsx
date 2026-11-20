import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { server } from "@/mocks/mock-server";
import * as api from "../api";
import { ProActivationProvider } from "../ProActivationProvider";
import { quote } from "./fixtures";

const route = vi.hoisted(() => ({
  search: new URLSearchParams(),
  replace: vi.fn(),
}));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: route.replace, push: vi.fn() }),
  useSearchParams: () => route.search,
}));
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

const paidReturn =
  "subscription=success&session_id=cs_test_1&plan=PRO&cycle=monthly";
// What the backend stores for a Checkout: the success URL it was given.
const successTemplate =
  "/onboarding?step=preparing&subscription=success&session_id={CHECKOUT_SESSION_ID}&plan=PRO&cycle=monthly";
const current = vi.fn();

function mount() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ProActivationProvider>
        <span>page</span>
      </ProActivationProvider>
    </QueryClientProvider>,
  );
}

function activationReady(returnTo: string) {
  server.use(
    http.get("*/api/proxy/api/credits/pro-activation/current", () => {
      current();
      return HttpResponse.json({
        ...quote,
        status: "ready",
        return_to: returnTo,
      });
    }),
    http.get("*/api/proxy/api/credits/trial", () =>
      HttpResponse.json({ active: false, converted: false }),
    ),
    http.get("*/api/proxy/api/credits/subscription", () =>
      HttpResponse.json({ tier: "PRO" }),
    ),
    http.get("*/api/proxy/api/chat/usage", () =>
      HttpResponse.json({ tier: "PRO" }),
    ),
  );
}

beforeEach(() => {
  sessionStorage.clear();
  current.mockReset();
  route.replace.mockReset();
  route.search = new URLSearchParams();
  window.history.replaceState({}, "", "/");
});
afterEach(() => vi.restoreAllMocks());

describe("Pro activation after a Stripe checkout return", () => {
  it("does not take a page reached in-app from a checkout return for one", () => {
    activationReady(successTemplate);
    // Onboarding replaced itself with /home: Next renders the new page before
    // it writes the new URL, so the browser still shows the checkout return.
    window.history.replaceState(
      {},
      "",
      `/onboarding?step=preparing&${paidReturn}`,
    );
    route.search = new URLSearchParams();
    const retrieve = vi.spyOn(api, "retrieveActivation");
    // render() flushes the mount effects, where a checkout return starts its lookup.
    mount();
    expect(retrieve).not.toHaveBeenCalled();
  });

  it("stays on the page Stripe returned to when the plan is ready", async () => {
    activationReady(
      "/settings/billing?subscription=success&session_id={CHECKOUT_SESSION_ID}&plan=PRO&cycle=monthly",
    );
    window.history.replaceState({}, "", `/settings/billing?${paidReturn}`);
    route.search = new URLSearchParams(paidReturn);
    mount();
    await screen.findByText("Pro is active");
    expect(current).toHaveBeenCalledTimes(1);
    expect(route.replace).not.toHaveBeenCalled();
  });

  it("returns a resumed upgrade to the page it started from", async () => {
    activationReady("/copilot/thread?resume=1#draft");
    sessionStorage.setItem(
      "pro-activation:user-1",
      JSON.stringify({ id: quote.id, token: quote.terms_token }),
    );
    mount();
    await screen.findByText("Pro is active");
    expect(route.replace).toHaveBeenCalledExactlyOnceWith(
      "/copilot/thread?resume=1#draft",
    );
  });
});

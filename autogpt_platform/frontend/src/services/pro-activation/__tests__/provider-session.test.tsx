import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { ProActivationProvider } from "../ProActivationProvider";
import { useProActivation } from "../useProActivation";
import { quote } from "./fixtures";

vi.mock("@/lib/auth/hooks/useAuthStore", async () => {
  const { create } = await import("zustand");
  return {
    useAuthStore: create<{ user: { id: string } | null }>(() => ({
      user: null,
    })),
  };
});

const userA = { id: "user-A", email: "a@example.com", user_metadata: {} };
const userB = { id: "user-B", email: "b@example.com", user_metadata: {} };

function Subject() {
  const activation = useProActivation();
  return (
    <>
      <input aria-label="Search agents" />
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

beforeEach(() => {
  useAuthStore.setState({ user: null });
  sessionStorage.clear();
  window.history.replaceState({}, "", "/");
  server.use(
    http.get(
      "*/api/proxy/api/credits/pro-activation/current",
      () => new HttpResponse(null, { status: 404 }),
    ),
  );
});

describe("Pro activation account lifecycle", () => {
  it("preserves page input and focus when the authenticated user hydrates", () => {
    mount();
    const input = screen.getByRole<HTMLInputElement>("textbox", {
      name: "Search agents",
    });
    input.focus();
    fireEvent.change(input, { target: { value: "My agent" } });

    act(() => useAuthStore.setState({ user: userA }));

    expect(screen.getByRole("textbox", { name: "Search agents" })).toBe(input);
    expect(input.value).toBe("My agent");
    expect(document.activeElement).toBe(input);
  });

  it("clears the previous account's activation readiness on account changes", async () => {
    useAuthStore.setState({ user: userA });
    sessionStorage.setItem(
      "pro-activation:user-A",
      JSON.stringify({ id: quote.id, token: quote.terms_token }),
    );
    server.use(
      http.get("*/api/proxy/api/credits/pro-activation/current", () =>
        HttpResponse.json({ ...quote, status: "ready" }),
      ),
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
    await screen.findByText("allowance ready");

    act(() => useAuthStore.setState({ user: userB }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByText("allowance waiting")).toBeDefined();
    expect(screen.queryByText("allowance ready")).toBeNull();
  });
});

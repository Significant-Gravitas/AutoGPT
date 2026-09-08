import { afterEach, beforeEach, describe, expect, test } from "vitest";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getGetV1GetAutoTopUpMockHandler } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";

import { WalletRefill } from "../WalletRefill";

beforeEach(() => {
  useAuthStore.setState({ user: { id: "user-a" } as User });
});

afterEach(() => {
  useAuthStore.setState({ user: null });
});

describe("WalletRefill", () => {
  test("renders the shared top-up form in the one-time top-up tab", async () => {
    server.use(getGetV1GetAutoTopUpMockHandler({ amount: 0, threshold: 0 }));

    render(<WalletRefill />);

    // The "top-up" tab is the default, so the extracted TopUpForm (with its
    // single "Amount" field) must render after the WalletRefill extraction.
    expect(await screen.findByLabelText("Amount")).toBeDefined();
    expect(screen.getByRole("button", { name: /top up/i })).toBeDefined();
  });

  test("resets the refill form and refetches config when identity changes", async () => {
    let configRequests = 0;
    server.use(
      getGetV1GetAutoTopUpMockHandler(() => {
        configRequests += 1;
        return { amount: 1000, threshold: 500 };
      }),
    );

    render(<WalletRefill />);
    const amount = await screen.findByLabelText("Amount");
    await waitFor(() => expect(configRequests).toBe(1));
    fireEvent.change(amount, { target: { value: "99" } });
    expect((amount as HTMLInputElement).value).toContain("99");

    act(() => {
      useAuthStore.setState({ user: { id: "user-b" } as User });
    });
    expect((screen.getByLabelText("Amount") as HTMLInputElement).value).toBe(
      "",
    );

    await waitFor(() => expect(configRequests).toBe(2));
  });
});

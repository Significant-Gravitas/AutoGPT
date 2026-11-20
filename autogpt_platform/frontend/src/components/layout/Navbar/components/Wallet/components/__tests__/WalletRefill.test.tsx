import { afterEach, beforeEach, describe, expect, test } from "vitest";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { server } from "@/mocks/mock-server";
import {
  getGetV1GetAutoTopUpMockHandler,
  getPostV1ConfigureAutoTopUpMockHandler,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";

import { WalletRefill } from "../WalletRefill";

const THRESHOLD = "Refill when balance drops below:";
const REFILL_AMOUNT = "Add this amount:";
const ENABLE_AUTO_REFILL = "Enable Auto-refill";

function fieldValue(label: string) {
  return (screen.getByLabelText(label) as HTMLInputElement).value;
}

function editField(label: string, value: string) {
  const input = screen.getByLabelText(label);
  fireEvent.change(input, { target: { value } });
  fireEvent.blur(input);
}

function enableAutoRefillButton() {
  return screen.getByRole("button", { name: ENABLE_AUTO_REFILL });
}

async function openTab(name: "One-time top up" | "Auto-refill") {
  await userEvent.click(screen.getByRole("tab", { name }));
}

async function openPrefilledAutoRefill() {
  await openTab("Auto-refill");
  await waitFor(() => expect(fieldValue(THRESHOLD)).toBe("$5"));
  expect(fieldValue(REFILL_AMOUNT)).toBe("$10");
}

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

  test("resets the one-time top-up amount and refetches config when identity changes", async () => {
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

  test("keeps the auto-refill draft across a tab switch", async () => {
    server.use(
      getGetV1GetAutoTopUpMockHandler({ amount: 1000, threshold: 500 }),
    );

    render(<WalletRefill />);
    await openPrefilledAutoRefill();
    editField(THRESHOLD, "20");
    editField(REFILL_AMOUNT, "40");

    await openTab("One-time top up");
    expect(screen.queryByLabelText(THRESHOLD)).toBeNull();
    await openTab("Auto-refill");

    expect(fieldValue(THRESHOLD)).toBe("$20");
    expect(fieldValue(REFILL_AMOUNT)).toBe("$40");
  });

  test("keeps Enable Auto-refill disabled across a tab switch while a save is pending", async () => {
    let configRequests = 0;
    let saves = 0;
    let releaseSave = () => {};
    const saveHeld = new Promise<void>((resolve) => {
      releaseSave = resolve;
    });
    server.use(
      getGetV1GetAutoTopUpMockHandler(() => {
        configRequests += 1;
        return { amount: 1000, threshold: 500 };
      }),
      getPostV1ConfigureAutoTopUpMockHandler(async () => {
        saves += 1;
        if (saves === 1) await saveHeld;
        return "Auto top-up config updated";
      }),
    );

    render(<WalletRefill />);
    await openPrefilledAutoRefill();

    try {
      await userEvent.click(enableAutoRefillButton());
      await waitFor(() => expect(saves).toBe(1));
      await waitFor(() =>
        expect(enableAutoRefillButton().hasAttribute("disabled")).toBe(true),
      );

      await openTab("One-time top up");
      await openTab("Auto-refill");

      expect(enableAutoRefillButton().hasAttribute("disabled")).toBe(true);
      await userEvent.click(enableAutoRefillButton());
    } finally {
      releaseSave();
    }

    await waitFor(() =>
      expect(enableAutoRefillButton().hasAttribute("disabled")).toBe(false),
    );
    await waitFor(() => expect(configRequests).toBe(2));
    expect(saves).toBe(1);
  });

  test("resets an edited auto-refill draft and loads the next account's config when identity changes", async () => {
    let releaseUserBConfig = () => {};
    const userBConfigHeld = new Promise<void>((resolve) => {
      releaseUserBConfig = resolve;
    });
    server.use(
      getGetV1GetAutoTopUpMockHandler(async () => {
        if (useAuthStore.getState().user?.id !== "user-b") {
          return { amount: 1000, threshold: 500 };
        }
        await userBConfigHeld;
        return { amount: 3000, threshold: 1500 };
      }),
    );

    render(<WalletRefill />);
    await openPrefilledAutoRefill();
    editField(THRESHOLD, "20");
    editField(REFILL_AMOUNT, "40");

    try {
      act(() => {
        useAuthStore.setState({ user: { id: "user-b" } as User });
      });
      await openTab("Auto-refill");
      expect(fieldValue(THRESHOLD)).toBe("");
      expect(fieldValue(REFILL_AMOUNT)).toBe("");
    } finally {
      releaseUserBConfig();
    }

    await waitFor(() => expect(fieldValue(THRESHOLD)).toBe("$15"));
    expect(fieldValue(REFILL_AMOUNT)).toBe("$30");
  });
});

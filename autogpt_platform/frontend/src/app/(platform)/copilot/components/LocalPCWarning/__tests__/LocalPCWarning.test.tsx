import { Key, storage } from "@/services/storage/local-storage";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { LocalPCWarning } from "../LocalPCWarning";

const auth = vi.hoisted(() => ({ userID: "user-1" }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: auth.userID } }),
}));

beforeEach(() => {
  auth.userID = "user-1";
  storage.clean(Key.COPILOT_LOCAL_PC_WARNING_ACKED);
});

afterEach(() => {
  storage.clean(Key.COPILOT_LOCAL_PC_WARNING_ACKED);
});

describe("LocalPCWarning", () => {
  test("shows the modal on first activation", async () => {
    render(<LocalPCWarning />);
    expect(
      await screen.findByText(/Code will run on your real machine/i),
    ).toBeDefined();
  });

  test("hides itself after acknowledgement and writes the flag", async () => {
    render(<LocalPCWarning />);
    const ack = await screen.findByRole("button", {
      name: /I understand — continue/i,
    });
    fireEvent.click(ack);
    expect(
      screen.queryByText(/Code will run on your real machine/i),
    ).toBeNull();
    expect(storage.get(Key.COPILOT_LOCAL_PC_WARNING_ACKED)).toBe("user-1");
  });

  test("does not render when previously acknowledged", () => {
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "user-1");
    render(<LocalPCWarning />);
    expect(
      screen.queryByText(/Code will run on your real machine/i),
    ).toBeNull();
  });

  test("requires acknowledgement when another user signs in", async () => {
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "user-1");
    const view = render(<LocalPCWarning />);
    auth.userID = "user-2";
    view.rerender(<LocalPCWarning />);

    expect(
      await screen.findByRole("button", { name: /I understand — continue/i }),
    ).toBeDefined();
  });

  test("does not accept the old browser-wide acknowledgement", async () => {
    storage.set(Key.COPILOT_LOCAL_PC_WARNING_ACKED, "true");
    render(<LocalPCWarning />);
    expect(
      await screen.findByRole("button", { name: /I understand — continue/i }),
    ).toBeDefined();
  });

  test("explains that the file root does not sandbox shell access", async () => {
    render(<LocalPCWarning />);
    expect(
      await screen.findByText(/folder you choose limits only/i),
    ).toBeDefined();
    expect(screen.getByText(/does not sandbox shell commands/i)).toBeDefined();
    expect(screen.getByText(/full user-level permissions/i)).toBeDefined();
    expect(screen.getByText(/autogpt-shim audit tail/)).toBeDefined();
  });

  test("can return to Cloud without acknowledging the warning", async () => {
    const onCancel = vi.fn();
    render(<LocalPCWarning onCancel={onCancel} />);

    fireEvent.click(
      await screen.findByRole("button", { name: "Use Cloud Instead" }),
    );

    expect(onCancel).toHaveBeenCalledOnce();
    expect(storage.get(Key.COPILOT_LOCAL_PC_WARNING_ACKED)).toBeNull();
  });
});

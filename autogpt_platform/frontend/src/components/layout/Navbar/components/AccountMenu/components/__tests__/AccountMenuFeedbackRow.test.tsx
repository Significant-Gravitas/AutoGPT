import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { AccountMenuFeedbackRow } from "../AccountMenuFeedbackRow";

const mockSidebar = vi.hoisted(() => ({
  current: null as null | {
    isMobile: boolean;
    setOpenMobile: (open: boolean) => void;
  },
}));

vi.mock("@/components/ui/sidebar", () => ({
  useOptionalSidebar: () => mockSidebar.current,
}));

const mockAuth = vi.hoisted(() => ({
  getCurrentUser: vi.fn(),
}));

const mockSentry = vi.hoisted(() => ({
  getReplay: vi.fn(),
}));

vi.mock("@/lib/auth/actions", () => mockAuth);
vi.mock("@sentry/nextjs", () => mockSentry);

async function renderRow() {
  render(<AccountMenuFeedbackRow />);
  await act(async () => {});
  return screen.getByRole("button", { name: "Give feedback" });
}

describe("AccountMenuFeedbackRow", () => {
  beforeEach(() => {
    mockSidebar.current = null;
    mockAuth.getCurrentUser.mockResolvedValue({
      user: { email: "user@example.com" },
    });
    mockSentry.getReplay.mockReturnValue({ getReplayId: () => "replay-123" });
    window.history.replaceState({}, "", "/library?sort=updatedAt");
  });

  test("renders a Give feedback button wired to the Tally feedback form", async () => {
    const button = await renderRow();

    expect(button.getAttribute("data-tally-open")).toBe("3yx2L0");
    expect(button.getAttribute("data-tally-emoji-text")).toBe("👋");
    expect(button.getAttribute("data-tally-emoji-animation")).toBe("wave");
    expect(button.getAttribute("data-sentry-replay-id")).toBe("replay-123");
    expect(button.getAttribute("data-sentry-replay-url")).toBe(
      "https://significant-gravitas.sentry.io/replays/replay-123/",
    );
    expect(button.getAttribute("data-page-url")).toBe(
      `${window.location.origin}/library`,
    );
    expect(button.getAttribute("data-is-authenticated")).toBe("true");
  });

  test("marks values that are not known yet", async () => {
    mockSentry.getReplay.mockReturnValue(undefined);
    mockAuth.getCurrentUser.mockReturnValue(new Promise(() => {}));
    const button = await renderRow();

    expect(button.getAttribute("data-sentry-replay-id")).toBe(
      "not-initialized",
    );
    expect(button.getAttribute("data-is-authenticated")).toBe("unknown");
  });

  test("closes the mobile sidebar sheet so the Tally popup is usable", async () => {
    const setOpenMobile = vi.fn();
    mockSidebar.current = { isMobile: true, setOpenMobile };

    fireEvent.click(await renderRow());

    expect(setOpenMobile).toHaveBeenCalledWith(false);
  });

  test("leaves the desktop sidebar alone", async () => {
    const setOpenMobile = vi.fn();
    mockSidebar.current = { isMobile: false, setOpenMobile };

    fireEvent.click(await renderRow());

    expect(setOpenMobile).not.toHaveBeenCalled();
  });

  test("works outside a sidebar", async () => {
    const button = await renderRow();
    expect(() => fireEvent.click(button)).not.toThrow();
  });
});

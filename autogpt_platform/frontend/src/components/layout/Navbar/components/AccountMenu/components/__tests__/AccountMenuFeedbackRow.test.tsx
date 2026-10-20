import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { stubTallyEmbed } from "@/tests/integrations/tally";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
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

const flushReplay = vi.fn();

let tally: ReturnType<typeof stubTallyEmbed>;

const mockUseAuth = vi.hoisted(() => ({
  current: { isLoggedIn: true, isUserLoading: false },
}));

vi.mock("@/lib/auth/actions", () => mockAuth);
vi.mock("@sentry/nextjs", () => mockSentry);
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => mockUseAuth.current,
}));

async function renderRow() {
  render(
    <>
      <AccountMenuFeedbackRow />
      <Toaster />
    </>,
  );
  await act(async () => {});
  return screen.getByRole("button", { name: "Give feedback" });
}

describe("AccountMenuFeedbackRow", () => {
  beforeEach(() => {
    mockSidebar.current = null;
    mockUseAuth.current = { isLoggedIn: true, isUserLoading: false };
    mockAuth.getCurrentUser.mockReset();
    flushReplay.mockReset();
    mockSentry.getReplay.mockReturnValue({
      getReplayId: () => "replay-123",
      flush: flushReplay,
    });
    window.history.replaceState({}, "", "/library?sort=updatedAt");
    tally = stubTallyEmbed();
  });

  afterEach(async () => {
    await tally.restore();
  });

  test("does not load the Tally script until someone asks for feedback", async () => {
    await renderRow();

    expect(tally.scripts).toHaveLength(0);
  });

  test("loads Tally on the first click, then opens the feedback form with the page context", async () => {
    fireEvent.click(await renderRow());

    expect(tally.scripts).toHaveLength(1);
    await tally.finishLoading();

    expect(tally.openPopup).toHaveBeenCalledTimes(1);
    expect(tally.openPopup).toHaveBeenCalledWith(
      "3yx2L0",
      expect.objectContaining({
        emoji: { text: "👋", animation: "wave" },
        hiddenFields: {
          sentryReplayId: "replay-123",
          sentryReplayUrl:
            "https://significant-gravitas.sentry.io/replays/replay-123/",
          pageUrl: `${window.location.origin}/library`,
          isAuthenticated: "true",
        },
      }),
    );
  });

  test("later clicks open the form without loading Tally again", async () => {
    const button = await renderRow();
    fireEvent.click(button);
    await tally.finishLoading();

    await act(async () => {
      fireEvent.click(button);
    });

    expect(tally.scripts).toHaveLength(1);
    expect(tally.openPopup).toHaveBeenCalledTimes(2);
  });

  test("flushes the Sentry replay once the form is open", async () => {
    fireEvent.click(await renderRow());
    await tally.finishLoading();

    act(() => tally.popupOptions().onOpen());

    expect(flushReplay).toHaveBeenCalledTimes(1);
  });

  test("marks values that are not known yet", async () => {
    mockSentry.getReplay.mockReturnValue(undefined);
    mockUseAuth.current = { isLoggedIn: false, isUserLoading: true };
    fireEvent.click(await renderRow());
    await tally.finishLoading();

    expect(tally.popupOptions().hiddenFields).toMatchObject({
      sentryReplayId: "not-initialized",
      sentryReplayUrl: "not-initialized",
      isAuthenticated: "unknown",
    });
  });

  test("offers a link to the form when Tally can't load, and tries again on the next click", async () => {
    const button = await renderRow();
    fireEvent.click(button);
    await tally.failLoading();

    expect(
      await screen.findByText("Couldn't open the feedback form"),
    ).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Open the form in a new tab" })
        .getAttribute("href"),
    ).toBe("https://tally.so/r/3yx2L0");

    fireEvent.click(button);
    expect(tally.scripts).toHaveLength(2);
    await tally.finishLoading();
    expect(tally.openPopup).toHaveBeenCalledTimes(1);
  });

  test("reads auth from the client store, not the getCurrentUser server action", async () => {
    await renderRow();

    expect(mockAuth.getCurrentUser).not.toHaveBeenCalled();
  });

  test("hides the decorative icon from assistive tech", async () => {
    const button = await renderRow();

    expect(button.querySelector("svg")?.getAttribute("aria-hidden")).toBe(
      "true",
    );
  });

  test("closes the mobile sidebar sheet so the Tally popup is usable", async () => {
    const setOpenMobile = vi.fn();
    mockSidebar.current = { isMobile: true, setOpenMobile };

    fireEvent.click(await renderRow());

    expect(setOpenMobile).toHaveBeenCalledWith(false);
    await tally.finishLoading();
    expect(tally.openPopup).toHaveBeenCalledTimes(1);
  });

  test("leaves the desktop sidebar alone", async () => {
    const setOpenMobile = vi.fn();
    mockSidebar.current = { isMobile: false, setOpenMobile };

    fireEvent.click(await renderRow());

    expect(setOpenMobile).not.toHaveBeenCalled();
    await tally.finishLoading();
    expect(tally.openPopup).toHaveBeenCalledTimes(1);
  });

  test("works outside a sidebar", async () => {
    fireEvent.click(await renderRow());

    expect(tally.scripts).toHaveLength(1);
    await tally.finishLoading();
    expect(tally.openPopup).toHaveBeenCalledTimes(1);
  });
});

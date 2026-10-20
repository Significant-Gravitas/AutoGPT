import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { stubTallyEmbed } from "@/tests/integrations/tally";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { FeedbackButton } from "../FeedbackButton";

vi.mock("@sentry/nextjs", () => ({ getReplay: vi.fn() }));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: false, isUserLoading: false }),
}));

let tally: ReturnType<typeof stubTallyEmbed>;

function feedbackButton() {
  return screen.getByRole<HTMLButtonElement>("button", {
    name: "Give Feedback",
  });
}

describe("FeedbackButton", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/marketplace");
    tally = stubTallyEmbed();
  });

  afterEach(() => {
    tally.restore();
  });

  test("loads Tally only after a click and waits for it before opening the form", async () => {
    render(<FeedbackButton />);
    expect(tally.scripts).toHaveLength(0);

    fireEvent.click(feedbackButton());

    expect(tally.scripts).toHaveLength(1);
    expect(feedbackButton().disabled).toBe(true);
    expect(tally.openPopup).not.toHaveBeenCalled();

    await tally.finishLoading();

    expect(feedbackButton().disabled).toBe(false);
    expect(tally.openPopup).toHaveBeenCalledWith(
      "3yx2L0",
      expect.objectContaining({
        hiddenFields: expect.objectContaining({ isAuthenticated: "false" }),
      }),
    );
  });

  test("hides while the form is open and comes back once it closes", async () => {
    render(<FeedbackButton />);
    fireEvent.click(feedbackButton());
    await tally.finishLoading();

    act(() => tally.popupOptions().onOpen());
    expect(screen.queryByRole("button", { name: "Give Feedback" })).toBeNull();

    act(() => tally.popupOptions().onClose());
    expect(feedbackButton()).toBeDefined();
  });
});

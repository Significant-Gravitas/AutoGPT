import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import SlackInstalledPage from "../page";

const mockUseSearchParams = vi.hoisted(() => vi.fn());
const mockReplace = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  usePathname: () => "/link/slack/installed",
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: mockReplace,
  }),
  useSearchParams: mockUseSearchParams,
}));

function setParams(params: Record<string, string>) {
  mockUseSearchParams.mockReturnValue(new URLSearchParams(params));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ shouldAdvanceTime: true });
  setParams({ team: "T1", app: "A1", bot: "UBOT" });
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { href: "http://localhost/link/slack/installed" },
  });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SlackInstalledPage", () => {
  test("opens the bot DM and then returns to the bots settings page", () => {
    render(<SlackInstalledPage />);

    expect(window.location.href).toBe("slack://user?team=T1&id=UBOT");
    expect(mockReplace).not.toHaveBeenCalled();

    vi.advanceTimersByTime(2000);

    expect(mockReplace).toHaveBeenCalledWith("/settings/bots");
  });

  test("still returns to settings when Slack gave us no bot user", () => {
    setParams({ team: "T1", app: "A1" });

    render(<SlackInstalledPage />);

    // No deep link to follow, so the page must not strand the user here.
    expect(window.location.href).toBe("http://localhost/link/slack/installed");
    vi.advanceTimersByTime(2000);
    expect(mockReplace).toHaveBeenCalledWith("/settings/bots");
  });

  test("offers a browser fallback for people without the Slack app", () => {
    render(<SlackInstalledPage />);

    expect(
      screen
        .getByRole("link", { name: /open slack in browser/i })
        .getAttribute("href"),
    ).toBe("https://slack.com/app_redirect?app=A1&team=T1");
  });

  test("retries the deep link when the user asks", () => {
    render(<SlackInstalledPage />);
    window.location.href = "http://localhost/link/slack/installed";

    fireEvent.click(screen.getByRole("button", { name: /^open slack$/i }));

    expect(window.location.href).toBe("slack://user?team=T1&id=UBOT");
  });
});

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";

let pathname = "/marketplace";

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => pathname,
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

import { CookieConsentBanner } from "../CookieConsentBanner";

describe("CookieConsentBanner", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  test("shows for a visitor without consent", async () => {
    pathname = "/marketplace";
    render(<CookieConsentBanner />);
    expect(await screen.findByText("We use cookies")).toBeDefined();
  });

  test("stays hidden on the public tour pages", async () => {
    pathname = "/tour/chat";
    render(<CookieConsentBanner />);
    // The banner renders after a consent-load effect; give it the same beat
    // the positive case needs before asserting absence.
    await Promise.resolve();
    expect(screen.queryByText("We use cookies")).toBeNull();
  });

  test("rejecting cookies clears the stored analytics identity", async () => {
    pathname = "/marketplace";
    // handleUpdateConsent reloads so the gates re-run; jsdom cannot navigate.
    const reload = vi.fn();
    Object.defineProperty(window, "location", {
      value: { ...window.location, reload },
      writable: true,
      configurable: true,
    });
    localStorage.setItem("agpt_anonymous_id", "visitor-1");
    localStorage.setItem("agpt_first_landing", '{"path":"/pricing"}');

    render(<CookieConsentBanner />);
    fireEvent.click(await screen.findByRole("button", { name: "Reject All" }));

    expect(localStorage.getItem("agpt_anonymous_id")).toBeNull();
    expect(localStorage.getItem("agpt_first_landing")).toBeNull();
    expect(reload).toHaveBeenCalled();
  });

  test("lets the visitor decide on advertising cookies separately", async () => {
    pathname = "/marketplace";
    render(<CookieConsentBanner />);
    fireEvent.click(await screen.findByRole("button", { name: "Settings" }));

    expect(
      await screen.findByLabelText("Toggle advertising cookies"),
    ).toBeDefined();
    expect(screen.getByText("Advertising")).toBeDefined();
  });
});

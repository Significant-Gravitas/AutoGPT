import { render, screen } from "@/tests/integrations/test-utils";
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
});

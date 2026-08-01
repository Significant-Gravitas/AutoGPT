import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { FeedbackButton } from "@/components/layout/Navbar/components/FeedbackButton";
import { getCurrentUser } from "@/lib/auth/actions";
import { TallyPopupProvider } from "../TallyPopup";

vi.mock("next/navigation", () => ({
  usePathname: () => "/library",
}));

vi.mock("@/lib/auth/actions", () => ({
  getCurrentUser: vi.fn(),
}));

vi.mock("@sentry/nextjs", () => ({
  getReplay: vi.fn(() => null),
}));

describe("TallyPopupProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    document
      .querySelectorAll('script[src="https://tally.so/widgets/embed.js"]')
      .forEach((script) => script.remove());
  });

  it("does not load Tally or inspect the user when feedback is disabled", () => {
    render(
      <TallyPopupProvider enabled={false}>
        <FeedbackButton />
      </TallyPopupProvider>,
    );

    expect(screen.queryByRole("button", { name: /give feedback/i })).toBeNull();
    expect(getCurrentUser).not.toHaveBeenCalled();
    expect(
      document.querySelector('script[src="https://tally.so/widgets/embed.js"]'),
    ).toBeNull();
  });
});

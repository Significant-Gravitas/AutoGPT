import {
  GMAIL,
  mockAccountDeletionAPI,
} from "@/tests/integrations/account-deletion-fixtures";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { DeleteAccountFlow } from "../DeleteAccountFlow";

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const viewport = vi.hoisted(() => ({ isLarge: true }));

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => (viewport.isLarge ? "lg" : "sm"),
  isLargeScreen: (breakpoint: string) => breakpoint === "lg",
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/settings/account",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "jane@example.com", user_metadata: {} },
    logOut: vi.fn(),
  }),
}));

const CLICKABLE = "button, a, input, textarea, select, label, [role='radio']";

// posthog-js autocapture drops an event when the clicked element or any
// ancestor has the ph-no-capture class. Scanning the whole body also catches
// anything the flow portals out of the dialog.
function getCapturableElements() {
  return Array.from(document.body.querySelectorAll(CLICKABLE)).filter(
    (element) => element.closest(".ph-no-capture") === null,
  );
}

beforeEach(() => {
  mockAccountDeletionAPI({ credentials: [GMAIL] });
});

describe.each([
  { layout: "modal", isLarge: true },
  { layout: "drawer", isLarge: false },
])("DeleteAccountFlow as a $layout", ({ isLarge }) => {
  test("keeps every click in both steps out of PostHog autocapture", async () => {
    viewport.isLarge = isLarge;
    render(<DeleteAccountFlow source="settings" open onOpenChange={vi.fn()} />);

    const reasonStep = await screen.findByRole("dialog", {
      name: "Why are you leaving?",
    });
    expect(reasonStep.hasAttribute("data-vaul-drawer")).toBe(!isLarge);
    expect(reasonStep.closest(".ph-no-capture")).not.toBeNull();
    fireEvent.click(
      within(reasonStep).getByRole("radio", {
        name: "I'm worried about my data or privacy",
      }),
    );
    await within(reasonStep).findByRole("button", {
      name: "Manage connected apps",
    });
    expect(getCapturableElements()).toEqual([]);

    fireEvent.click(within(reasonStep).getByRole("button", { name: "Skip" }));

    const confirmStep = await screen.findByRole("dialog", {
      name: "Delete your account?",
    });
    expect(
      within(confirmStep).getByRole("button", { name: "Delete account" }),
    ).toBeDefined();
    expect(getCapturableElements()).toEqual([]);
  });
});

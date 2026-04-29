import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useOnboardingWizardStore } from "../../store";
import { SubscriptionStep } from "../SubscriptionStep/SubscriptionStep";

vi.mock("@/components/atoms/FadeIn/FadeIn", () => ({
  FadeIn: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

vi.mock("@/components/atoms/AutoGPTLogo/AutoGPTLogo", () => ({
  AutoGPTLogo: () => <span>AutoGPTLogo</span>,
}));

afterEach(cleanup);

beforeEach(() => {
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().goToStep(4);
});

describe("SubscriptionStep", () => {
  test("renders the three plan cards by display name", () => {
    render(<SubscriptionStep />);
    expect(screen.getByRole("heading", { name: /^Pro$/ })).toBeDefined();
    expect(screen.getByRole("heading", { name: /^Max$/ })).toBeDefined();
    expect(screen.getByRole("heading", { name: /^Team$/ })).toBeDefined();
  });

  test("monthly USD prices render with two decimals", () => {
    render(<SubscriptionStep />);
    expect(screen.getByText(/\$50\.00/)).toBeDefined();
    expect(screen.getByText(/\$320\.00/)).toBeDefined();
  });

  test("toggling yearly billing updates the store and shows /year", () => {
    render(<SubscriptionStep />);
    fireEvent.click(screen.getByRole("button", { name: /Yearly billing/i }));
    expect(useOnboardingWizardStore.getState().selectedBilling).toBe("yearly");
    expect(screen.getAllByText(/\/ year/i).length).toBeGreaterThan(0);
  });

  test("selecting Pro persists selectedPlan and advances to step 5", () => {
    render(<SubscriptionStep />);
    fireEvent.click(screen.getByRole("button", { name: /Get Pro/i }));
    const state = useOnboardingWizardStore.getState();
    expect(state.selectedPlan).toBe("PRO");
    expect(state.currentStep).toBe(5);
  });

  test("selecting Team opens the intake form and does not advance", () => {
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    try {
      render(<SubscriptionStep />);
      fireEvent.click(screen.getByRole("button", { name: /Contact sales/i }));
      expect(openSpy).toHaveBeenCalledWith(
        expect.stringContaining("tally.so"),
        "_blank",
        "noopener,noreferrer",
      );
      const state = useOnboardingWizardStore.getState();
      expect(state.selectedPlan).toBeNull();
      expect(state.currentStep).toBe(4);
    } finally {
      openSpy.mockRestore();
    }
  });

  test("changing country persists the country code", () => {
    render(<SubscriptionStep />);
    fireEvent.click(screen.getByRole("button", { name: /United States/i }));
    fireEvent.click(screen.getByRole("button", { name: /European Union/i }));
    expect(useOnboardingWizardStore.getState().selectedCountryCode).toBe("EU");
  });
});

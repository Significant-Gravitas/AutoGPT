import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useOnboardingWizardStore } from "../../store";
import { RoleStep } from "../RoleStep";

vi.mock("@/components/atoms/Emoji/Emoji", () => ({
  Emoji: ({ text }: { text: string }) => <span>{text}</span>,
}));

vi.mock("@/components/atoms/FadeIn/FadeIn", () => ({
  FadeIn: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

beforeEach(() => {
  vi.useFakeTimers();
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().setName("Alice");
  useOnboardingWizardStore.getState().goToStep(2);
});

describe("RoleStep", () => {
  test("renders all role cards", () => {
    render(<RoleStep />);

    expect(screen.getByText("Founder / CEO")).toBeDefined();
    expect(screen.getByText("Operations")).toBeDefined();
    expect(screen.getByText("Sales / BD")).toBeDefined();
    expect(screen.getByText("Marketing")).toBeDefined();
    expect(screen.getByText("Product / PM")).toBeDefined();
    expect(screen.getByText("Engineering")).toBeDefined();
    expect(screen.getByText("HR / People")).toBeDefined();
    expect(screen.getByText("Other")).toBeDefined();
  });

  test("displays the user name in the heading", () => {
    render(<RoleStep />);

    expect(
      screen.getAllByText(/what best describes you, alice/i).length,
    ).toBeGreaterThan(0);
  });

  test("selecting a non-Other role auto-advances after delay", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /engineering/i }));

    expect(useOnboardingWizardStore.getState().role).toBe("Engineering");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);

    vi.advanceTimersByTime(350);

    expect(useOnboardingWizardStore.getState().currentStep).toBe(3);
  });

  test("selecting 'Other' does not auto-advance", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /\bother\b/i }));

    vi.advanceTimersByTime(500);

    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  test("selecting 'Other' shows text input and Continue button", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /\bother\b/i }));

    expect(screen.getByPlaceholderText(/describe your role/i)).toBeDefined();
    expect(screen.getByRole("button", { name: /continue/i })).toBeDefined();
  });

  test("Continue button is disabled when Other input is empty", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /\bother\b/i }));

    const continueBtn = screen.getByRole("button", { name: /continue/i });
    expect(continueBtn.hasAttribute("disabled")).toBe(true);
  });

  test("Continue button advances when Other role text is filled", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /\bother\b/i }));
    fireEvent.change(screen.getByPlaceholderText(/describe your role/i), {
      target: { value: "Designer" },
    });

    const continueBtn = screen.getByRole("button", { name: /continue/i });
    expect(continueBtn.hasAttribute("disabled")).toBe(false);

    fireEvent.click(continueBtn);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(3);
  });

  test("switching from Other to a regular role cancels Other and auto-advances", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /\bother\b/i }));
    expect(screen.getByPlaceholderText(/describe your role/i)).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /marketing/i }));

    expect(useOnboardingWizardStore.getState().role).toBe("Marketing");
    vi.advanceTimersByTime(350);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(3);
  });
});

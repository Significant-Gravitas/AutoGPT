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
  useOnboardingWizardStore.getState().goToStep(1);
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

  test("asks what best describes the user", () => {
    render(<RoleStep />);

    expect(
      screen.getByRole("heading", { name: "What best describes you?" }),
    ).toBeDefined();
  });

  test("selecting a role does not advance on its own", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /engineering/i }));

    expect(useOnboardingWizardStore.getState().role).toBe("Engineering");
    vi.advanceTimersByTime(1000);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
  });

  test("Next is disabled until a role is chosen, then advances", () => {
    render(<RoleStep />);

    const next = screen.getByRole("button", { name: "Next" });
    expect(next.hasAttribute("disabled")).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: /engineering/i }));
    expect(next.hasAttribute("disabled")).toBe(false);

    fireEvent.click(next);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  test("selecting 'Other' shows the text input", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /\bother\b/i }));

    expect(screen.getByPlaceholderText(/describe your role/i)).toBeDefined();
  });

  test("Next stays disabled for Other until the role is described", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /\bother\b/i }));
    const next = screen.getByRole("button", { name: "Next" });
    expect(next.hasAttribute("disabled")).toBe(true);

    fireEvent.change(screen.getByPlaceholderText(/describe your role/i), {
      target: { value: "Designer" },
    });
    expect(next.hasAttribute("disabled")).toBe(false);

    fireEvent.click(next);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  test("switching from Other to a regular role hides the input and keeps the step", () => {
    render(<RoleStep />);

    fireEvent.click(screen.getByRole("button", { name: /\bother\b/i }));
    expect(screen.getByPlaceholderText(/describe your role/i)).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /marketing/i }));

    expect(useOnboardingWizardStore.getState().role).toBe("Marketing");
    vi.advanceTimersByTime(1000);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
  });
});

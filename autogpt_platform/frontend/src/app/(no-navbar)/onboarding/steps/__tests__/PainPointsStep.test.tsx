import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useOnboardingWizardStore } from "../../store";
import { PainPointsStep } from "../PainPointsStep";

vi.mock("@/components/atoms/Emoji/Emoji", () => ({
  Emoji: ({ text }: { text: string }) => <span>{text}</span>,
}));

vi.mock("@/components/atoms/FadeIn/FadeIn", () => ({
  FadeIn: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

function getCard(name: RegExp) {
  return screen.getByRole("button", { name });
}

function clickCard(name: RegExp) {
  fireEvent.click(getCard(name));
}

function getLaunchButton() {
  return screen.getByRole("button", { name: /continue/i });
}

afterEach(cleanup);

beforeEach(() => {
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().setName("Alice");
  useOnboardingWizardStore.getState().setRole("Founder/CEO");
  useOnboardingWizardStore.getState().goToStep(3);
});

describe("PainPointsStep", () => {
  test("renders all pain point cards", () => {
    render(<PainPointsStep />);

    expect(getCard(/finding leads/i)).toBeDefined();
    expect(getCard(/email & outreach/i)).toBeDefined();
    expect(getCard(/reports & data/i)).toBeDefined();
    expect(getCard(/customer support/i)).toBeDefined();
    expect(getCard(/social media/i)).toBeDefined();
    expect(getCard(/something else/i)).toBeDefined();
  });

  test("shows default helper text", () => {
    render(<PainPointsStep />);

    expect(
      screen.getAllByText(/pick up to 3 to start/i).length,
    ).toBeGreaterThan(0);
  });

  test("selecting a card marks it as pressed", () => {
    render(<PainPointsStep />);

    clickCard(/finding leads/i);

    expect(getCard(/finding leads/i).getAttribute("aria-pressed")).toBe("true");
  });

  test("launch button is disabled when nothing is selected", () => {
    render(<PainPointsStep />);

    expect(getLaunchButton().hasAttribute("disabled")).toBe(true);
  });

  test("launch button is enabled after selecting a pain point", () => {
    render(<PainPointsStep />);

    clickCard(/finding leads/i);

    expect(getLaunchButton().hasAttribute("disabled")).toBe(false);
  });

  test("shows success text when 3 items are selected", () => {
    render(<PainPointsStep />);

    clickCard(/finding leads/i);
    clickCard(/email & outreach/i);
    clickCard(/reports & data/i);

    expect(screen.getAllByText(/3 selected/i).length).toBeGreaterThan(0);
  });

  test("does not select a 4th item when at the limit", () => {
    render(<PainPointsStep />);

    clickCard(/finding leads/i);
    clickCard(/email & outreach/i);
    clickCard(/reports & data/i);
    clickCard(/customer support/i);

    expect(getCard(/customer support/i).getAttribute("aria-pressed")).toBe(
      "false",
    );
  });

  test("can deselect when at the limit and select a different one", () => {
    render(<PainPointsStep />);

    clickCard(/finding leads/i);
    clickCard(/email & outreach/i);
    clickCard(/reports & data/i);

    clickCard(/finding leads/i);
    expect(getCard(/finding leads/i).getAttribute("aria-pressed")).toBe(
      "false",
    );

    clickCard(/customer support/i);
    expect(getCard(/customer support/i).getAttribute("aria-pressed")).toBe(
      "true",
    );
  });

  test("shows input when 'Something else' is selected", () => {
    render(<PainPointsStep />);

    clickCard(/something else/i);

    expect(
      screen.getByPlaceholderText(/what else takes up your time/i),
    ).toBeDefined();
  });

  test("launch button is disabled when 'Something else' selected but input empty", () => {
    render(<PainPointsStep />);

    clickCard(/something else/i);

    expect(getLaunchButton().hasAttribute("disabled")).toBe(true);
  });

  test("launch button is enabled when 'Something else' selected and input filled", () => {
    render(<PainPointsStep />);

    clickCard(/something else/i);
    fireEvent.change(
      screen.getByPlaceholderText(/what else takes up your time/i),
      { target: { value: "Manual invoicing" } },
    );

    expect(getLaunchButton().hasAttribute("disabled")).toBe(false);
  });
});

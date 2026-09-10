import { beforeEach, describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { useOnboardingWizardStore } from "../../../store";
import { IntroStep } from "../IntroStep";
import { INTRO_SLIDES } from "../helpers";

beforeEach(() => useOnboardingWizardStore.getState().reset());

describe("IntroStep", () => {
  it("shows the team slide and advances the wizard on Next", () => {
    render(<IntroStep slide="team" />);
    expect(
      screen.getByRole("heading", { name: INTRO_SLIDES.team.title }),
    ).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  it("shows the AutoPilot slide", () => {
    render(<IntroStep slide="autopilot" />);
    expect(
      screen.getByRole("heading", { name: INTRO_SLIDES.autopilot.title }),
    ).toBeDefined();
  });
});

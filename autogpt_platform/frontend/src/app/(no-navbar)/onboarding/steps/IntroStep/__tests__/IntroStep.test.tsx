import { beforeEach, describe, expect, it } from "vitest";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { useOnboardingWizardStore } from "../../../store";
import { IntroStep } from "../IntroStep";
import { INTRO_SLIDES } from "../helpers";
import { TEAM } from "../components/teamSceneHelpers";

beforeEach(() => useOnboardingWizardStore.getState().reset());

describe("IntroStep", () => {
  it("welcomes the user on the team slide and moves on to Otto", () => {
    render(<IntroStep slide="team" />);
    expect(screen.getByText("Welcome to AutoGPT")).toBeDefined();
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Your new team, ready on day one",
      }),
    ).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Meet Otto" }));
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  it("shows the Otto slide with its own Next button", () => {
    render(<IntroStep slide="autopilot" />);
    expect(
      screen.getByRole("heading", { name: INTRO_SLIDES.autopilot.title }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Next" })).toBeDefined();
    expect(screen.queryByText("Welcome to AutoGPT")).toBeNull();
  });

  it("lists every member of the team with their role", () => {
    render(<IntroStep slide="team" />);
    const team = screen.getByRole("list", { name: "Your team" });
    const members = within(team).getAllByRole("listitem");
    expect(members).toHaveLength(TEAM.length);
    expect(within(members[0]).getByText("Otto")).toBeDefined();
    expect(within(members[0]).getByText("Head of AI")).toBeDefined();
    expect(within(team).getByText("Maria")).toBeDefined();
    expect(within(team).getByText("Marketing")).toBeDefined();
    expect(within(team).getByText("Devon")).toBeDefined();
    expect(within(team).getByText("Security")).toBeDefined();
  });

  it("shows the team without invented task states", () => {
    render(<IntroStep slide="team" />);
    expect(screen.queryAllByTestId("status-dot")).toHaveLength(0);
  });
});

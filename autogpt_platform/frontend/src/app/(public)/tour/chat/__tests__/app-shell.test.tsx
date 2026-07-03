import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";

// DotDistortionShader paints a canvas/WebGL frame that happy-dom cannot run and
// that is purely decorative — stub it so the real chat tree can render.
vi.mock("@/components/ui/dot-distortion-shader", () => ({
  DotDistortionShader: () => null,
}));

vi.mock("@/app/(platform)/copilot/useIsMobile", () => ({
  useIsMobile: () => false,
}));

const flagState = vi.hoisted(() => ({ tourAppShell: true }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.TOUR_APP_SHELL ? flagState.tourAppShell : false,
  };
});

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import TourChatPage from "../page";
import { DEFAULT_SCENARIO_ID } from "../script/tourScenarios";
import { useTourStore } from "../tourStore";

function getSendBar() {
  return screen.getByRole("button", { name: /^Send:/i });
}

const ADVANCE_STEP_MS = 200;
// Longest turn is ~7.7s of parts — including the 5s fake run — plus the 3s
// hold before the demo completes.
const ADVANCE_TOTAL_MS = 13000;

// The prompt bar is prefilled and locked — the visitor only presses Enter to
// send. Timers advance in small chunks so effects that register new timers
// mid-stream get picked up (see main.test.tsx for the full rationale).
async function pressEnterToSend() {
  fireEvent.keyDown(getSendBar(), { key: "Enter" });
  for (
    let elapsed = 0;
    elapsed < ADVANCE_TOTAL_MS;
    elapsed += ADVANCE_STEP_MS
  ) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ADVANCE_STEP_MS);
    });
  }
}

describe("Tour chat app shell (tour-app-shell flag)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    flagState.tourAppShell = true;
    // Both stores are module-level state — reset between tests.
    useTourStore.setState({ activeScenarioId: DEFAULT_SCENARIO_ID });
    useCopilotUIStore.getState().clearArtifactPreview();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  test("renders the sidebar with scenarios as chat sessions and only Marketplace enabled", () => {
    render(<TourChatPage />);

    // Scenario pills are replaced by sidebar chat sessions.
    expect(document.querySelector("[aria-pressed]")).toBeNull();
    expect(screen.getByText("Recent chats")).toBeDefined();
    for (const label of [
      "Daily brief",
      "Call prep",
      "Competitor watch",
      "Support queue",
    ]) {
      expect(screen.getByRole("button", { name: label })).toBeDefined();
    }

    // Marketplace is the only live navigation target.
    const marketplace = screen.getByRole("link", { name: "Marketplace" });
    expect(marketplace.getAttribute("href")).toBe("/marketplace");
    for (const label of ["New Task", "Search", "Agents", "Build", "Files"]) {
      const item = screen.getByRole("button", { name: label });
      expect(item.getAttribute("aria-disabled")).toBe("true");
    }
  });

  test("clicking a sidebar session switches the demo scenario", async () => {
    render(<TourChatPage />);

    fireEvent.click(screen.getByRole("button", { name: "Daily brief" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(
      screen.getByText(/pull my unread emails and calendar/i),
    ).toBeDefined();
  });

  test("finishing the demo opens the artifact panel with the mock markdown file", async () => {
    render(<TourChatPage />);

    await pressEnterToSend();
    await pressEnterToSend();

    // The real copilot ArtifactPanel opens with the scenario's mock file.
    expect(
      await screen.findByText("competitor-pricing-report.md"),
    ).toBeDefined();
    expect(useCopilotUIStore.getState().artifactPanel.activeArtifact?.id).toBe(
      "tour-competitor-watch",
    );
  });

  test("flag off keeps the scenario pills layout", () => {
    flagState.tourAppShell = false;
    render(<TourChatPage />);

    expect(
      screen
        .getByRole("button", { name: "Competitor watch" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    expect(screen.queryByText("Recent chats")).toBeNull();
  });
});

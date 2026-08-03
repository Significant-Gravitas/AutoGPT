import { getGetBrainDumpStatusMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { server } from "@/mocks/mock-server";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PreparingStep } from "../PreparingStep";

const INTRO_PATH_KEY = "autogpt:onboarding-intro-path";

const GENERIC_CHECKLIST = [
  "Personalizing your experience",
  "Connecting automation engines",
  "Building your space",
];
const DUMP_CHECKLIST = [
  "Reading your brain dump",
  "Briefing AutoPilot on your work",
  "Building your space",
  "Finding tools for your work",
];

beforeEach(() => {
  window.sessionStorage.removeItem(INTRO_PATH_KEY);
  // The dump path polls for pipeline status once the bar fills.
  server.use(
    getGetBrainDumpStatusMockHandler200(() => ({
      status: "completed" as const,
      greeting_ready: true,
    })),
  );
});

afterEach(() => {
  cleanup();
});

describe("PreparingStep checklist copy", () => {
  it("shows the brain-dump checklist on the dump path (intro path A, flag on)", async () => {
    setIntroPath("A");

    render(<PreparingStep onComplete={vi.fn()} isBrainDumpEnabled />);

    for (const item of DUMP_CHECKLIST) {
      expect(await screen.findByText(item)).toBeDefined();
    }
    expect(screen.queryByText("Personalizing your experience")).toBeNull();
    expect(screen.queryByText("Connecting automation engines")).toBeNull();
  });

  it("shows the generic checklist on the skip path (intro path B, flag on)", async () => {
    setIntroPath("B");

    render(<PreparingStep onComplete={vi.fn()} isBrainDumpEnabled />);

    for (const item of GENERIC_CHECKLIST) {
      expect(await screen.findByText(item)).toBeDefined();
    }
    expect(screen.queryByText("Reading your brain dump")).toBeNull();
    expect(screen.queryByText("Briefing AutoPilot on your work")).toBeNull();
  });

  it("shows the generic checklist when the brain-dump flag is off, even with a stale path A", async () => {
    setIntroPath("A");

    render(<PreparingStep onComplete={vi.fn()} isBrainDumpEnabled={false} />);

    for (const item of GENERIC_CHECKLIST) {
      expect(await screen.findByText(item)).toBeDefined();
    }
    expect(screen.queryByText("Reading your brain dump")).toBeNull();
  });

  it("shows the generic checklist when no intro path was recorded", async () => {
    render(<PreparingStep onComplete={vi.fn()} isBrainDumpEnabled />);

    for (const item of GENERIC_CHECKLIST) {
      expect(await screen.findByText(item)).toBeDefined();
    }
    expect(screen.queryByText("Reading your brain dump")).toBeNull();
  });
});

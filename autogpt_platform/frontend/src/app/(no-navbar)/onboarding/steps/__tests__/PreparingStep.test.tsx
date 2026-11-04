import {
  getGetBrainDumpRecommendedProvidersMockHandler200,
  getGetBrainDumpStatusMockHandler200,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { server } from "@/mocks/mock-server";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import {
  act,
  cleanup,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
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
  "Briefing Otto on your work",
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
  vi.useRealTimers();
});

describe("PreparingStep checklist copy", () => {
  it("keeps the last step in progress while recommendations are pending", async () => {
    vi.useFakeTimers();
    setIntroPath("A");
    server.use(
      getGetBrainDumpRecommendedProvidersMockHandler200({
        ready: false,
        providers: [],
      }),
    );
    const onComplete = vi.fn();
    render(<PreparingStep onComplete={onComplete} />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_500);
    });

    expect(screen.getAllByText("Completed")).toHaveLength(3);
    const lastStep = screen.getAllByRole("listitem").at(-1)!;
    expect(within(lastStep).getByText("In progress")).toBeDefined();
    expect(onComplete).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4_500);
    });
    expect(screen.getAllByText("Completed")).toHaveLength(4);
    expect(onComplete).toHaveBeenCalledOnce();
  });

  it("announces each step's state and completes it only after its time elapses", async () => {
    vi.useFakeTimers();
    setIntroPath("B");
    const onComplete = vi.fn();
    render(<PreparingStep onComplete={onComplete} />);
    const steps = screen.getAllByRole("listitem");

    expect(within(steps[0]).getByText("In progress")).toBeDefined();
    expect(within(steps[1]).getByText("Pending")).toBeDefined();
    expect(within(steps[2]).getByText("Pending")).toBeDefined();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(300);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(50);
    });
    expect(within(steps[0]).getByText("In progress")).toBeDefined();
    expect(screen.queryByText("Completed")).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_350);
    });
    expect(within(steps[0]).getByText("Completed")).toBeDefined();
    expect(within(steps[1]).getByText("In progress")).toBeDefined();
    expect(within(steps[2]).getByText("Pending")).toBeDefined();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_550);
    });
    expect(within(steps[1]).getByText("Completed")).toBeDefined();
    expect(within(steps[2]).getByText("In progress")).toBeDefined();
    expect(onComplete).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(50);
    });

    expect(onComplete).toHaveBeenCalledOnce();
    expect(screen.getAllByText("Completed")).toHaveLength(3);
    for (const item of GENERIC_CHECKLIST) {
      expect(screen.getByText(item)).toBeDefined();
    }
  });

  it("shows the brain-dump checklist on the dump path (intro path A)", async () => {
    setIntroPath("A");

    render(<PreparingStep onComplete={vi.fn()} />);

    for (const item of DUMP_CHECKLIST) {
      expect(await screen.findByText(item)).toBeDefined();
    }
    expect(screen.queryByText("Personalizing your experience")).toBeNull();
    expect(screen.queryByText("Connecting automation engines")).toBeNull();
  });

  it("shows the generic checklist on the skip path (intro path B)", async () => {
    setIntroPath("B");

    render(<PreparingStep onComplete={vi.fn()} />);

    for (const item of GENERIC_CHECKLIST) {
      expect(await screen.findByText(item)).toBeDefined();
    }
    expect(screen.queryByText("Reading your brain dump")).toBeNull();
    expect(screen.queryByText("Briefing Otto on your work")).toBeNull();
  });

  it("shows the generic checklist when no intro path was recorded", async () => {
    render(<PreparingStep onComplete={vi.fn()} />);

    for (const item of GENERIC_CHECKLIST) {
      expect(await screen.findByText(item)).toBeDefined();
    }
    expect(screen.queryByText("Reading your brain dump")).toBeNull();
  });
});

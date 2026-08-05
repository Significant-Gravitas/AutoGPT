import {
  getDiscardBrainDumpMockHandler200,
  getFinalizeBrainDumpMockHandler200,
  getGetBrainDumpStatusMockHandler200,
  getUploadBrainDumpPartMockHandler200,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import OnboardingPage from "../page";
import { NO_PAYWALL_STEPS, PAYWALL_FIRST_STEPS } from "../store";
import { useOnboardingWizardStore } from "../store";

// IndexedDB does not exist in happy-dom and `fake-indexeddb` is not a
// devDependency here, so the recording store is replaced with an
// in-memory double. Everything above it (the recorder, the upload queue,
// the recovery prompt) runs for real.
const { recordingStoreState } = vi.hoisted(() => ({
  recordingStoreState: {
    // Ids the store was asked to mark finalized, kept separately because
    // `clearRecording` deletes the row moments later.
    finalizedIds: [] as string[],
    meta: null as {
      key: string;
      recordingId: string;
      mimeType: string;
      startedAt: number;
      durationSecs: number;
      finalized: boolean;
    } | null,
    parts: [] as {
      id: string;
      recordingId: string;
      partIndex: number;
      blob: Blob;
      savedAt: number;
      uploaded: boolean;
    }[],
  },
}));

vi.mock("../steps/BrainDumpStep/recordingStore", () => ({
  isIndexedDBAvailable: () => true,
  partId: (recordingId: string, partIndex: number) =>
    `${recordingId}:${partIndex}`,
  savePart: async (part: (typeof recordingStoreState.parts)[number]) => {
    recordingStoreState.parts = [
      ...recordingStoreState.parts.filter((p) => p.id !== part.id),
      part,
    ];
  },
  markPartUploaded: async (id: string) => {
    recordingStoreState.parts = recordingStoreState.parts.map((p) =>
      p.id === id ? { ...p, uploaded: true } : p,
    );
  },
  getParts: async (recordingId: string) =>
    recordingStoreState.parts
      .filter((p) => p.recordingId === recordingId)
      .sort((a, b) => a.partIndex - b.partIndex),
  saveMeta: async (meta: Record<string, unknown>) => {
    if (meta.finalized) {
      recordingStoreState.finalizedIds.push(String(meta.recordingId));
    }
    recordingStoreState.meta = {
      ...meta,
      key: "current",
    } as typeof recordingStoreState.meta;
  },
  getMeta: async () => recordingStoreState.meta,
  getMetaById: async (recordingId: string) =>
    recordingStoreState.meta?.recordingId === recordingId
      ? recordingStoreState.meta
      : null,
  clearRecording: async () => {
    recordingStoreState.parts = [];
    recordingStoreState.meta = null;
  },
}));

vi.mock("posthog-js", () => ({
  default: { capture: vi.fn() },
}));

vi.mock("../steps/WelcomeStep", () => ({
  WelcomeStep: () => <div data-testid="step-welcome" />,
}));
vi.mock("../steps/RoleStep", () => ({
  RoleStep: () => <div data-testid="step-role" />,
}));
vi.mock("../steps/SubscriptionStep/SubscriptionStep", () => ({
  SubscriptionStep: () => <div data-testid="step-subscription" />,
}));
vi.mock("../steps/PreparingStep", () => ({
  PreparingStep: () => <div data-testid="step-preparing" />,
}));

let currentSearchParams = new URLSearchParams();
const routerReplace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: routerReplace,
    push: vi.fn(),
    refresh: vi.fn(),
  }),
  useSearchParams: () => currentSearchParams,
  usePathname: () => "/onboarding",
}));

const mockRefreshSession = vi.fn(() => Promise.resolve({ user: null }));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isLoggedIn: true,
    isUserLoading: false,
    user: null,
    refreshSession: mockRefreshSession,
  }),
}));

vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1OnboardingState: () =>
    Promise.resolve({ status: 200, data: { completedSteps: [] } }),
  getV1CheckIfOnboardingIsCompleted: () =>
    Promise.resolve({ status: 200, data: false }),
  patchV1UpdateOnboardingState: () => Promise.resolve({ status: 200 }),
  postV1CompleteOnboardingStep: () => Promise.resolve({ status: 200 }),
  postV1SubmitOnboardingProfile: () => Promise.resolve({ status: 200 }),
}));

vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: (opts: {
    query: { select: (res: { status: number; data: unknown }) => unknown };
  }) => ({
    data: opts.query.select({ status: 200, data: { tier: "NO_TIER" } }),
    isLoading: false,
  }),
}));

vi.mock("@/app/api/helpers", () => ({
  resolveResponse: (p: Promise<{ data: unknown }>) => p.then((r) => r.data),
}));

// Flag-aware so the payment flag and the brain-dump flag can be driven
// independently — they gate different step slots.
let mockFlags: Record<string, boolean> = {};
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ENABLE_PLATFORM_PAYMENT: "enable-platform-payment",
    ONBOARDING_BRAIN_DUMP: "onboarding-brain-dump",
  },
  useGetFlag: (flag: string) => mockFlags[flag] ?? false,
}));

vi.mock("launchdarkly-react-client-sdk", () => ({
  useLDClient: () => ({
    waitForInitialization: () => Promise.resolve(),
  }),
}));

const STEP_STORAGE_KEY = "autogpt:onboarding-highest-step";
const INTRO_PATH_KEY = "autogpt:onboarding-intro-path";
const PILLBOX_HEADING = "What's eating your time?";
const DUMP_HEADLINE = "What keeps stealing your week?";

class FakeMediaRecorder {
  static isTypeSupported() {
    return true;
  }
  state: "inactive" | "recording" = "inactive";
  mimeType: string;
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;

  constructor(_stream: unknown, options?: { mimeType?: string }) {
    this.mimeType = options?.mimeType ?? "audio/webm";
  }

  start(_timesliceMs?: number) {
    this.state = "recording";
    this.ondataavailable?.({
      data: new Blob(["chunk-one"], { type: this.mimeType }),
    });
  }

  stop() {
    this.state = "inactive";
    this.onstop?.();
  }
}

const getUserMedia = vi.fn();

function installBrowserRecordingAPIs() {
  Object.defineProperty(globalThis, "MediaRecorder", {
    configurable: true,
    writable: true,
    value: FakeMediaRecorder,
  });
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    writable: true,
    value: { getUserMedia },
  });
}

// Records every brain-dump call so "flag off makes no brain-dump request"
// can be asserted, and so request bodies can be inspected.
function recordBrainDumpTraffic() {
  const calls: { endpoint: string; body?: unknown }[] = [];
  server.use(
    getFinalizeBrainDumpMockHandler200(async (info) => {
      calls.push({ endpoint: "finalize", body: await info.request.json() });
      return { status: "completed" as const, input_mode: "voice" as const };
    }),
    getUploadBrainDumpPartMockHandler200(() => {
      calls.push({ endpoint: "parts" });
      return {
        recording_id: "r",
        part_index: 0,
        received_bytes: 9,
        total_bytes: 9,
      };
    }),
    getGetBrainDumpStatusMockHandler200(() => {
      calls.push({ endpoint: "status" });
      return { status: "completed" as const, greeting_ready: true };
    }),
    getDiscardBrainDumpMockHandler200(() => {
      calls.push({ endpoint: "discard" });
      return { status: null };
    }),
    http.get(
      "http://localhost:3000/api/proxy/api/onboarding/brain-dump/recording",
      () => {
        calls.push({ endpoint: "recording" });
        return HttpResponse.json({});
      },
    ),
  );
  return calls;
}

function finalizeReturns(response: {
  status: "completed" | "failed";
  error_code?: string;
}) {
  const bodies: unknown[] = [];
  server.use(
    getFinalizeBrainDumpMockHandler200(async (info) => {
      bodies.push(await info.request.json());
      return { ...response, input_mode: "voice" as const };
    }),
  );
  return bodies;
}

function stepDots(container: HTMLElement) {
  return container.querySelectorAll("div.h-2.rounded-full").length;
}

function progressWidth(container: HTMLElement) {
  const bar = container.querySelector<HTMLElement>("div.bg-purple-400");
  return bar?.style.width ?? null;
}

// Lands the wizard on the step the pillboxes / the brain dump share.
function landOnPainPointsStep() {
  window.sessionStorage.setItem(
    STEP_STORAGE_KEY,
    String(NO_PAYWALL_STEPS.painPoints),
  );
  currentSearchParams = new URLSearchParams(
    `step=${NO_PAYWALL_STEPS.painPoints}`,
  );
}

beforeEach(() => {
  installBrowserRecordingAPIs();
  getUserMedia.mockReset();
  getUserMedia.mockResolvedValue({ getTracks: () => [] });
  mockFlags = {};
  currentSearchParams = new URLSearchParams();
  routerReplace.mockClear();
  recordingStoreState.meta = null;
  recordingStoreState.parts = [];
  recordingStoreState.finalizedIds = [];
  useOnboardingWizardStore.getState().reset();
  window.sessionStorage.removeItem(STEP_STORAGE_KEY);
  window.sessionStorage.removeItem(INTRO_PATH_KEY);
});

afterEach(() => {
  cleanup();
});

describe("onboarding brain dump — flag gating", () => {
  it("renders the brain dump in the pillbox slot when the flag is on", async () => {
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);

    expect(await screen.findByText(DUMP_HEADLINE)).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Start recording" }),
    ).toBeDefined();
    expect(screen.queryByText(PILLBOX_HEADING)).toBeNull();
  });

  it("leaves the pillboxes untouched and makes no brain-dump request when the flag is off", async () => {
    const calls = recordBrainDumpTraffic();
    mockFlags = {};
    landOnPainPointsStep();

    render(<OnboardingPage />);

    expect(await screen.findByText(PILLBOX_HEADING)).toBeDefined();
    expect(
      screen.getByText("Pick the tasks you'd love to hand off to AutoPilot"),
    ).toBeDefined();
    expect(screen.queryByText(DUMP_HEADLINE)).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Start recording" }),
    ).toBeNull();
    expect(screen.queryByText("Skip for now")).toBeNull();

    // Give any stray effect a chance to fire before declaring silence.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(calls).toEqual([]);
  });

  // Guards the assertion above: proves the recorder above is actually wired
  // to the brain-dump endpoints, so an empty `calls` means silence rather
  // than a handler that never intercepts anything.
  it("records brain-dump traffic when the flag is on and the user acts", async () => {
    const calls = recordBrainDumpTraffic();
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Skip for now" }));

    await waitFor(() =>
      expect(calls.map((c) => c.endpoint)).toEqual(["finalize"]),
    );
  });
});

describe("onboarding brain dump — typed fallback", () => {
  it("opens the typed composer under the same headline when mic permission is denied", async () => {
    getUserMedia.mockRejectedValue(
      new DOMException("denied", "NotAllowedError"),
    );
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    await userEvent.click(
      screen.getByRole("button", { name: "Start recording" }),
    );

    expect(
      await screen.findByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
    ).toBeDefined();
    // Same headline, not a dead end.
    expect(screen.getByText(DUMP_HEADLINE)).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Start recording" }),
    ).toBeNull();
    // Offering a way back to the orb would be a dead end here: the browser
    // has already refused the microphone.
    expect(screen.queryByRole("button", { name: "record instead" })).toBeNull();
  });

  it("opens the typed composer from the rest state via 'type instead'", async () => {
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    expect(
      screen.queryByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
    ).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "type instead" }));

    expect(
      await screen.findByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
    ).toBeDefined();
    expect(getUserMedia).not.toHaveBeenCalled();
  });
});

describe("onboarding brain dump — skip", () => {
  it("posts input_mode 'skipped' and advances the wizard", async () => {
    const bodies = finalizeReturns({ status: "completed" });
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    await userEvent.click(screen.getByRole("button", { name: "Skip for now" }));

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({
      recording_id: expect.any(String),
      input_mode: "skipped",
    });

    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
  });
});

describe("onboarding brain dump — finishing a take", () => {
  // The success path all the way through: without it the finalize
  // bookkeeping could throw and be swallowed by its own try/catch and no
  // test would notice.
  it("marks the take finalized and clears it before advancing", async () => {
    const partUploads: string[] = [];
    server.use(
      getUploadBrainDumpPartMockHandler200(() => {
        partUploads.push("part");
        return {
          recording_id: "r",
          part_index: 0,
          received_bytes: 9,
          total_bytes: 9,
        };
      }),
    );
    const bodies = finalizeReturns({ status: "completed" });
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    await userEvent.click(
      screen.getByRole("button", { name: "Start recording" }),
    );
    await waitFor(() => expect(partUploads).toHaveLength(1));
    const doneButtons = await screen.findAllByRole("button", {
      name: "I'm done",
    });
    await userEvent.click(doneButtons[doneButtons.length - 1]);

    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(bodies).toHaveLength(1);
    const recordingId = (bodies[0] as { recording_id: string }).recording_id;
    // Marked finalized by id, then cleared: an unfinalized row left behind
    // makes the next visit offer back a take that is already submitted.
    expect(recordingStoreState.finalizedIds).toEqual([recordingId]);
    expect(recordingStoreState.meta).toBeNull();
    expect(recordingStoreState.parts).toEqual([]);
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  // Skipping mid-submit ran a second `nextStep()` behind the finalize
  // already in flight, landing past the last step on a blank screen with
  // Back and Log out hidden.
  it("takes 'Skip for now' away while the dump is being submitted", async () => {
    const partUploads: string[] = [];
    server.use(
      getUploadBrainDumpPartMockHandler200(() => {
        partUploads.push("part");
        return {
          recording_id: "r",
          part_index: 0,
          received_bytes: 9,
          total_bytes: 9,
        };
      }),
      getFinalizeBrainDumpMockHandler200(async () => {
        await new Promise(() => undefined);
        return { status: "completed" as const, input_mode: "voice" as const };
      }),
    );
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    expect(screen.getByRole("button", { name: "Skip for now" })).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: "Start recording" }),
    );
    await waitFor(() => expect(partUploads).toHaveLength(1));
    const doneButtons = await screen.findAllByRole("button", {
      name: "I'm done",
    });
    await userEvent.click(doneButtons[doneButtons.length - 1]);

    expect(await screen.findByText("Got it. One second…")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
  });
});

describe("onboarding brain dump — recovery", () => {
  it("offers to pick up an unfinalized recording that still has parts", async () => {
    recordingStoreState.meta = {
      key: "current",
      recordingId: "rec-1",
      mimeType: "audio/webm",
      startedAt: Date.now(),
      durationSecs: 95,
      finalized: false,
    };
    recordingStoreState.parts = [
      {
        id: "rec-1:0",
        recordingId: "rec-1",
        partIndex: 0,
        blob: new Blob(["chunk"], { type: "audio/webm" }),
        savedAt: Date.now(),
        uploaded: false,
      },
    ];
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);

    expect(
      await screen.findByText("Pick up where you left off?"),
    ).toBeDefined();
    // The duration sits in its own <span>, so match on the paragraph's
    // combined text rather than a single text node.
    expect(
      screen.getByText(
        (_content, element) =>
          element?.textContent === "We kept the 1:35 you already recorded.",
        { selector: "p" },
      ),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Use that recording" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Start over" })).toBeDefined();
  });

  it("does not offer recovery when the stored recording has no parts", async () => {
    recordingStoreState.meta = {
      key: "current",
      recordingId: "rec-empty",
      mimeType: "audio/webm",
      startedAt: Date.now(),
      durationSecs: 12,
      finalized: false,
    };
    recordingStoreState.parts = [];
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);

    expect(await screen.findByText(DUMP_HEADLINE)).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText("Pick up where you left off?")).toBeNull(),
    );
    expect(
      screen.getByRole("button", { name: "Start recording" }),
    ).toBeDefined();
  });
});

describe("onboarding brain dump — failure", () => {
  it("keeps the copy calm and offers retry plus download when finalize reports failure", async () => {
    const partUploads: string[] = [];
    server.use(
      getUploadBrainDumpPartMockHandler200(() => {
        partUploads.push("part");
        return {
          recording_id: "r",
          part_index: 0,
          received_bytes: 9,
          total_bytes: 9,
        };
      }),
    );
    const finalizeBodies = finalizeReturns({
      status: "failed",
      error_code: "transcription_error",
    });
    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    await userEvent.click(
      screen.getByRole("button", { name: "Start recording" }),
    );
    // Wait for the first chunk to reach the server so "I'm done" is not
    // racing the upload queue.
    await waitFor(() => expect(partUploads).toHaveLength(1));

    const doneButtons = await screen.findAllByRole("button", {
      name: "I'm done",
    });
    await userEvent.click(doneButtons[doneButtons.length - 1]);

    expect(await screen.findByText("That didn't go through.")).toBeDefined();
    // The failure has to come from finalize reporting `failed`, not from the
    // upload queue giving up before it ever got there.
    expect(finalizeBodies).toHaveLength(1);
    expect(finalizeBodies[0]).toMatchObject({
      input_mode: "voice",
      mime_type: "audio/webm",
      recording_id: expect.any(String),
    });
    expect(
      screen.getByText("Your recording is safe. Try again."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Download recording" }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Continue without it" }),
    ).toBeDefined();
    // The wizard must not advance on failure.
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });
});

describe("onboarding step map integrity", () => {
  it("keeps the step constants identical regardless of the brain-dump flag", () => {
    expect(PAYWALL_FIRST_STEPS).toEqual({
      subscription: 1,
      welcome: 2,
      role: 3,
      painPoints: 4,
      preparing: 5,
    });
    expect(NO_PAYWALL_STEPS).toEqual({
      welcome: 1,
      role: 2,
      painPoints: 3,
      preparing: 4,
    });
  });

  it("renders the same screen count and progress with the flag on and off", async () => {
    mockFlags = {};
    landOnPainPointsStep();
    const off = render(<OnboardingPage />);
    await screen.findByText(PILLBOX_HEADING);
    const offDots = stepDots(off.container);
    const offWidth = progressWidth(off.container);
    const offUrl = routerReplace.mock.calls.map((c) => c[0]);

    cleanup();
    routerReplace.mockClear();
    useOnboardingWizardStore.getState().reset();

    mockFlags = { "onboarding-brain-dump": true };
    landOnPainPointsStep();
    const on = render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    const onDots = stepDots(on.container);
    const onWidth = progressWidth(on.container);
    const onUrl = routerReplace.mock.calls.map((c) => c[0]);

    expect(offDots).toBe(3);
    expect(offWidth).toBe("75%");
    expect(onDots).toBe(offDots);
    expect(onWidth).toBe(offWidth);
    expect(onUrl).toEqual(offUrl);
  });
});

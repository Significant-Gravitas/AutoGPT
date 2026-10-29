import {
  getDiscardBrainDumpMockHandler200,
  getFinalizeBrainDumpMockHandler200,
  getGetBrainDumpStatusMockHandler200,
  getUploadBrainDumpPartMockHandler200,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import openapi from "@/app/api/openapi.json";
import { server } from "@/mocks/mock-server";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import posthog from "posthog-js";
import type { ReactNode } from "react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  onTestFinished,
  vi,
} from "vitest";
import OnboardingPage from "../page";
import { BRAIN_DUMP_PROMPT_SETS } from "../steps/BrainDumpStep/prompts";
import { NO_PAYWALL_STEPS, PAYWALL_LAST_STEPS } from "../store";
import { useOnboardingWizardStore } from "../store";
import { makeProgress } from "./progress-fixture";

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

// Each wizard save, in order, and whether Hire was already on screen.
const { wizardSaves } = vi.hoisted(() => ({
  wizardSaves: [] as { role: string; hireShown: boolean }[],
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
  clearSkippedRecordings: async () => {
    recordingStoreState.parts = [];
    recordingStoreState.meta = null;
  },
}));

// These tests navigate by NO_PAYWALL_STEPS, which is the layout for a
// deployment with neither a paywall nor a self-host connect step. isLocal()
// is true for anything not explicitly CLOUD, including the test environment,
// so it is pinned here rather than left to the default.
vi.mock("@/services/environment", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: { ...actual.environment, isLocal: () => false },
  };
});

vi.mock("posthog-js", () => ({
  default: { capture: vi.fn() },
}));

// AnimatePresence can retain the outgoing timer in happy-dom even after the
// recovery controls render. These flow tests assert the current headline and
// actions independently of the decorative exit animation.
vi.mock("@/components/atoms/SwapFade/SwapFade", () => ({
  SwapFade: function SwapFade({
    children,
    className,
  }: {
    children: ReactNode;
    className?: string;
  }) {
    return <div className={className}>{children}</div>;
  },
}));

vi.mock("../steps/RoleStep", () => ({
  RoleStep: () => <div data-testid="step-role" />,
}));
vi.mock("../steps/SubscriptionStep/SubscriptionStep", () => ({
  SubscriptionStep: () => <div data-testid="step-subscription" />,
}));
vi.mock("../steps/HireStep/HireStep", () => ({
  HireStep: () => <div data-testid="step-hire" />,
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
    user: { id: "brain-user", email: "brain@example.com", user_metadata: {} },
    refreshSession: mockRefreshSession,
  }),
}));

vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1OnboardingState: () =>
    Promise.resolve({
      status: 200,
      data: {
        userId: "brain-user",
        completedSteps: [],
        wizardRevision: 0,
        wizardProgress: makeProgress({
          currentStep: "painPoints",
          completedSteps: ["team", "autopilot", "role"],
        }),
      },
    }),
  getV1CheckIfOnboardingIsCompleted: () =>
    Promise.resolve({ status: 200, data: false }),
  patchV1UpdateOnboardingState: (body: {
    wizardProgress: { role: string };
  }) => {
    wizardSaves.push({
      role: body.wizardProgress.role,
      hireShown: document.querySelector('[data-testid="step-hire"]') !== null,
    });
    return Promise.resolve({
      status: 200,
      data: { userId: "brain-user", wizardRevision: 1 },
    });
  },
  postV1CompleteOnboardingStep: () => Promise.resolve({ status: 200 }),
  postV1SubmitOnboardingProfile: () => Promise.resolve({ status: 200 }),
}));

vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  getGetSubscriptionStatusQueryKey: () => ["/api/credits/subscription"],
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
  },
  useGetFlag: (flag: string) => mockFlags[flag] ?? false,
  useFlagStatus: (flag: string) => ({
    enabled: mockFlags[flag] ?? false,
    ready: true,
  }),
}));

const STEP_STORAGE_KEY = "autogpt:onboarding-highest-step";
const INTRO_PATH_KEY = "autogpt:onboarding-intro-path";
const PILLBOX_HEADING = "What's eating your time?";
const DUMP_HEADLINE = "Talk to me about your work";
const TYPING_HEADLINE = "Write to me about your work";
const REST_GUIDANCE =
  "Tell me what you do and what eats up your week. Otto uses it to pick your experts and plan your first week. A minute is plenty, and you can type instead.";

// Copy split across a nested emphasis: matches the element that holds all
// of it, not each piece.
function fullText(text: string) {
  return (_content: string, element: Element | null) =>
    element?.textContent === text &&
    Array.from(element.children).every((child) => child.textContent !== text);
}

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

class FakeAudioContext {
  state: AudioContextState = "running";
  sampleRate = 48_000;

  createAnalyser() {
    return {
      fftSize: 256,
      frequencyBinCount: 128,
      smoothingTimeConstant: 0,
      getByteTimeDomainData(samples: Uint8Array) {
        samples.fill(128);
      },
      getByteFrequencyData(samples: Uint8Array) {
        samples.fill(0);
      },
    } as unknown as AnalyserNode;
  }

  createMediaStreamSource() {
    return {
      connect() {},
      disconnect() {},
    } as unknown as MediaStreamAudioSourceNode;
  }

  resume() {
    return Promise.resolve();
  }

  close() {
    return Promise.resolve();
  }
}

const getUserMedia = vi.fn();

function installBrowserRecordingAPIs() {
  Object.defineProperty(globalThis, "MediaRecorder", {
    configurable: true,
    writable: true,
    value: FakeMediaRecorder,
  });
  Object.defineProperty(globalThis, "AudioContext", {
    configurable: true,
    writable: true,
    value: FakeAudioContext,
  });
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true,
    writable: true,
    value: { getUserMedia },
  });
}

// Records every brain-dump call so request payloads
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
    // Answered by default so the preparing step's team gate never stalls a
    // wizard test; the section itself is exercised on the greeting page.
    http.get(
      "http://localhost:3000/api/proxy/api/onboarding/brain-dump/recommended-experts",
      () => HttpResponse.json({ ready: true, team: null }),
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

// A voice take is judged too thin, and the skip's record goes unanswered
// until `release()`, so any step change that waits on it never happens.
function holdSkipRecord() {
  const skipBodies: unknown[] = [];
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  server.use(
    getFinalizeBrainDumpMockHandler200(async (info) => {
      const body = (await info.request.json()) as { input_mode: string };
      if (body.input_mode !== "skipped") {
        return {
          status: "failed" as const,
          error_code: "insufficient_content",
          input_mode: "voice" as const,
        };
      }
      skipBodies.push(body);
      await held;
      return { status: "completed" as const, input_mode: "skipped" as const };
    }),
  );
  return { skipBodies, release };
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
  wizardSaves.length = 0;
  useOnboardingWizardStore.getState().reset();
  localStorage.clear();
  window.sessionStorage.removeItem(STEP_STORAGE_KEY);
  window.sessionStorage.removeItem(INTRO_PATH_KEY);
});

afterEach(() => {
  cleanup();
});

describe("onboarding brain dump", () => {
  it("explains what to share before choosing Talk or Write", async () => {
    landOnPainPointsStep();

    render(<OnboardingPage />);

    expect(await screen.findByText(fullText(REST_GUIDANCE))).toBeDefined();
    // The middle sentence is the reason to share, so it stands out.
    expect(
      screen.getByText(
        "Otto uses it to pick your experts and plan your first week.",
      ).tagName,
    ).toBe("SPAN");
    expect(screen.getByRole("button", { name: "Talk" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Write" })).toBeDefined();
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("renders the brain dump as the personalization step", async () => {
    landOnPainPointsStep();

    render(<OnboardingPage />);

    expect(await screen.findByText(DUMP_HEADLINE)).toBeDefined();
    expect(screen.getByRole("button", { name: "Talk" })).toBeDefined();
    expect(screen.queryByText(PILLBOX_HEADING)).toBeNull();
  });

  it("renders Otto as the visual, waiting at rest", async () => {
    landOnPainPointsStep();

    render(<OnboardingPage />);

    const avatar = await screen.findByTestId("autopilot-avatar");
    expect(avatar.dataset.screen).toBe("rest");
    expect(screen.getByRole("button", { name: "Talk" })).toBeDefined();
  });

  // Guards the assertion above: proves the recorder above is actually wired
  // to the brain-dump endpoints, so an empty `calls` means silence rather
  // than a handler that never intercepts anything.
  it("records brain-dump traffic when the user acts", async () => {
    const calls = recordBrainDumpTraffic();
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
  it("opens the typed composer under a writing headline when mic permission is denied", async () => {
    getUserMedia.mockRejectedValue(
      new DOMException("denied", "NotAllowedError"),
    );
    // happy-dom reports every permission as granted, which reads as the
    // system refusing an allowed site.
    Object.defineProperty(navigator, "permissions", {
      configurable: true,
      value: {
        query: vi.fn().mockResolvedValue(new FakePermissionStatus("denied")),
      },
    });
    onTestFinished(() => {
      Reflect.deleteProperty(navigator, "permissions");
    });
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    await userEvent.click(screen.getByRole("button", { name: "Talk" }));

    expect(
      await screen.findByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
    ).toBeDefined();
    // The headline turns to writing; the step itself is not a dead end.
    expect(screen.getByText(TYPING_HEADLINE)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Talk" })).toBeNull();
    // The way back stays: it asks the browser for the microphone again.
    expect(
      screen.getByRole("button", { name: "record instead" }),
    ).toBeDefined();
    expect(screen.getByRole("alert").textContent).toContain(
      "Your browser blocked the microphone, so write here instead.",
    );
  });

  // No microphone, or one another app holds: Talk used to do nothing at all.
  it.each([
    new DOMException("", "NotFoundError"),
    new DOMException("", "NotReadableError"),
    new TypeError("navigator.mediaDevices is undefined"),
  ])(
    "says the microphone couldn't open and offers Write when it fails with $name",
    async (failure) => {
      vi.mocked(posthog.capture).mockClear();
      getUserMedia.mockRejectedValue(failure);
      landOnPainPointsStep();

      render(<OnboardingPage />);
      await screen.findByText(DUMP_HEADLINE);
      await userEvent.click(screen.getByRole("button", { name: "Talk" }));

      expect(
        await screen.findByText(
          "We couldn't open your microphone. You can write it instead.",
        ),
      ).toBeDefined();
      expect(screen.getByRole("button", { name: "Write" })).toBeDefined();
      // Unlike a denial, this can be fixed on the device, so Talk stays.
      expect(screen.getByRole("button", { name: "Talk" })).toBeDefined();
      expect(posthog.capture).toHaveBeenCalledWith("brain_dump_mic_failed", {
        error_name: failure.name,
      });
      expect(
        vi.mocked(posthog.capture).mock.calls.map(([event]) => event),
      ).not.toContain("brain_dump_permission_denied");
    },
  );

  it("opens the typed composer from the rest state via Write", async () => {
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    expect(
      screen.queryByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
    ).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Write" }));

    expect(
      await screen.findByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
    ).toBeDefined();
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("caps a long paste at the backend's limit and says how much is left", async () => {
    const backendLimit = finalizeTextMaxLength();
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Write" }));
    const box = await screen.findByPlaceholderText<HTMLTextAreaElement>(
      "What repeats every week? What would you hand off first?",
    );

    await userEvent.click(box);
    await userEvent.paste("a".repeat(backendLimit + 50));

    expect(box.value).toHaveLength(backendLimit);
    expect(screen.getByText("0 characters left")).toBeDefined();
  });

  it("returns to the text with the limit spelled out when finalize refuses it as too long", async () => {
    server.use(
      http.post(
        "http://localhost:3000/api/proxy/api/onboarding/brain-dump/finalize",
        () =>
          HttpResponse.json(
            {
              detail: [
                {
                  type: "string_too_long",
                  loc: ["body", "text"],
                  msg: "String should have at most 20000 characters",
                  ctx: { max_length: 20000 },
                },
              ],
            },
            { status: 422 },
          ),
      ),
    );
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Write" }));
    await userEvent.type(
      await screen.findByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
      "Invoices every Monday",
    );
    await userEvent.click(screen.getByRole("button", { name: "I'm done" }));

    expect(
      await screen.findByText(
        "That's over the 20,000-character limit; trim it and try again.",
      ),
    ).toBeDefined();
    expect(
      screen.getByPlaceholderText<HTMLTextAreaElement>(
        "What repeats every week? What would you hand off first?",
      ).value,
    ).toBe("Invoices every Monday");
    expect(screen.queryByText("Your recording is safe. Try again.")).toBeNull();

    await userEvent.type(
      screen.getByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
      "{Backspace}",
    );

    expect(
      screen.queryByText(
        "That's over the 20,000-character limit; trim it and try again.",
      ),
    ).toBeNull();
  });
});

describe("onboarding brain dump — skip", () => {
  it("posts input_mode 'skipped' and advances the wizard", async () => {
    const bodies = finalizeReturns({ status: "completed" });
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    await userEvent.click(screen.getByRole("button", { name: "Skip for now" }));

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({
      recording_id: expect.any(String),
      input_mode: "skipped",
    });

    expect(await screen.findByTestId("step-hire")).toBeDefined();
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
  });

  // SECRT-2982: while the first skip waited on its record the button stayed
  // up, the second click recorded a second skip, and both advanced past Hire.
  it("moves to Hire on the click, once, however often it is clicked", async () => {
    const skip = holdSkipRecord();
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    await userEvent.dblClick(
      screen.getByRole("button", { name: "Skip for now" }),
    );

    expect(await screen.findByTestId("step-hire")).toBeDefined();
    await waitFor(() => expect(skip.skipBodies).toHaveLength(1));
    skip.release();
    // Room for a second skip's advance to land, had one been sent.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(skip.skipBodies).toHaveLength(1);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(
      NO_PAYWALL_STEPS.hire,
    );
  });
});

// The server builds the hire step's team from the role in the saved progress,
// and the wizard saves a change 200 ms after it is made. Picking the role and
// leaving the step in one go leaves no time for that timer.
describe("onboarding brain dump — the role reaches the server first", () => {
  it("saves a role picked just before a skip before showing Hire", async () => {
    finalizeReturns({ status: "completed" });
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    act(() => useOnboardingWizardStore.getState().setRole("Marketing"));
    fireEvent.click(screen.getByRole("button", { name: "Skip for now" }));

    expect(await screen.findByTestId("step-hire")).toBeDefined();
    await waitFor(() =>
      expect(wizardSaves.map((save) => save.role)).toContain("Marketing"),
    );
    expect(wizardSaves.find((save) => save.role === "Marketing")).toEqual({
      role: "Marketing",
      hireShown: false,
    });
  });

  it.each([
    ["typed", sendTypedTake],
    ["recorded", sendRecordedTake],
  ])(
    "saves a role picked just before a %s take before sending the take",
    async (_label, sendTake) => {
      let rolesSavedAtFinalize: string[] = [];
      server.use(
        getUploadBrainDumpPartMockHandler200(() => ({
          recording_id: "r",
          part_index: 0,
          received_bytes: 9,
          total_bytes: 9,
        })),
        getFinalizeBrainDumpMockHandler200(() => {
          rolesSavedAtFinalize = wizardSaves.map((save) => save.role);
          return { status: "completed" as const, input_mode: "voice" as const };
        }),
      );
      landOnPainPointsStep();

      render(<OnboardingPage />);
      await screen.findByText(DUMP_HEADLINE);
      await sendTake(() =>
        act(() => useOnboardingWizardStore.getState().setRole("Marketing")),
      );

      expect(await screen.findByTestId("step-hire")).toBeDefined();
      expect(rolesSavedAtFinalize).toContain("Marketing");
    },
  );
});

// Each picks the role (`pickRole`) and sends in one go, with no time between.
async function sendTypedTake(pickRole: () => void) {
  await userEvent.click(screen.getByRole("button", { name: "Write" }));
  await userEvent.type(
    await screen.findByPlaceholderText(
      "What repeats every week? What would you hand off first?",
    ),
    "I run a bakery",
  );
  pickRole();
  fireEvent.click(screen.getByRole("button", { name: "I'm done" }));
}

async function sendRecordedTake(pickRole: () => void) {
  await userEvent.click(screen.getByRole("button", { name: "Talk" }));
  const send = await screen.findByRole("button", { name: "Send recording" });
  await waitFor(() =>
    expect(recordingStoreState.parts.map((part) => part.uploaded)).toEqual([
      true,
    ]),
  );
  pickRole();
  fireEvent.click(send);
}

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
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    await userEvent.click(screen.getByRole("button", { name: "Talk" }));
    await waitFor(() =>
      expect(screen.getByTestId("autopilot-avatar").dataset.screen).toBe(
        "recording",
      ),
    );
    await waitFor(() => expect(partUploads).toHaveLength(1));
    await userEvent.click(
      await screen.findByRole("button", { name: "Send recording" }),
    );

    expect(await screen.findByTestId("step-hire")).toBeDefined();
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
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    expect(screen.getByRole("button", { name: "Skip for now" })).toBeDefined();

    await userEvent.click(screen.getByRole("button", { name: "Talk" }));
    await waitFor(() => expect(partUploads).toHaveLength(1));
    await userEvent.click(
      await screen.findByRole("button", { name: "Send recording" }),
    );

    expect(await screen.findByText("Got it. One second…")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
  });

  it("shows immediate progress while canceling a recording", async () => {
    let finishDiscard: (() => void) | undefined;
    recordBrainDumpTraffic();
    server.use(
      getDiscardBrainDumpMockHandler200(async () => {
        await new Promise<void>((resolve) => {
          finishDiscard = resolve;
        });
        return { status: null };
      }),
    );
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Talk" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Cancel recording" }),
    );

    expect(await screen.findByText("Discard recording?")).toBeDefined();
    expect(
      screen.getByText(/This permanently deletes your current take/),
    ).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: "Keep recording" }),
    );
    expect(screen.queryByText("Discard recording?")).toBeNull();
    expect(finishDiscard).toBeUndefined();

    await userEvent.click(
      screen.getByRole("button", { name: "Cancel recording" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Discard recording" }),
    );

    expect(screen.getByTestId("recording-feedback-slot")).toBeDefined();
    const cancelingButton = (await screen.findByRole("button", {
      name: "Canceling recording",
    })) as HTMLButtonElement;
    expect(cancelingButton.getAttribute("aria-busy")).toBe("true");
    expect(await screen.findByTestId("recording-control-loader")).toBeDefined();
    expect(await screen.findByText("Discarding this take…")).toBeDefined();
    expect(
      (
        screen.getByRole("button", {
          name: "Send recording",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
    expect(
      screen
        .getByRole("button", { name: "Send recording" })
        .getAttribute("aria-disabled"),
    ).toBe("true");
    expect(
      (
        screen.getByRole("button", {
          name: "Retry recording",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
    expect(
      screen
        .getByRole("button", { name: "Retry recording" })
        .getAttribute("aria-disabled"),
    ).toBe("true");
    expect(cancelingButton.disabled).toBe(false);
    expect(cancelingButton.getAttribute("aria-disabled")).toBe("true");
    await waitFor(() => expect(document.activeElement).toBe(cancelingButton));
    await userEvent.click(cancelingButton);
    expect(screen.queryByText("Discard recording?")).toBeNull();

    await waitFor(() => expect(finishDiscard).toBeDefined());
    finishDiscard!();
    expect(await screen.findByRole("button", { name: "Talk" })).toBeDefined();
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
    landOnPainPointsStep();

    render(<OnboardingPage />);

    expect(await screen.findByText(DUMP_HEADLINE)).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText("Pick up where you left off?")).toBeNull(),
    );
    expect(screen.getByRole("button", { name: "Talk" })).toBeDefined();
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
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    await userEvent.click(screen.getByRole("button", { name: "Talk" }));
    // Wait for the first chunk to reach the server so sending is not
    // racing the upload queue.
    await waitFor(() => expect(partUploads).toHaveLength(1));

    await userEvent.click(
      await screen.findByRole("button", { name: "Send recording" }),
    );

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
    expect(screen.queryByTestId("step-hire")).toBeNull();
  });

  it("retries a typed take with its text and sends no recording", async () => {
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
    const responses = [
      { status: "failed" as const, error_code: "quality_check_unavailable" },
      { status: "completed" as const },
    ];
    const finalizeBodies: unknown[] = [];
    server.use(
      getFinalizeBrainDumpMockHandler200(async (info) => {
        finalizeBodies.push(await info.request.json());
        return {
          ...responses[finalizeBodies.length - 1],
          input_mode: "typed" as const,
        };
      }),
    );
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Write" }));
    await userEvent.type(
      await screen.findByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
      "I run a bakery",
    );
    await userEvent.click(screen.getByRole("button", { name: "I'm done" }));

    expect(
      await screen.findByText("Your text is safe. Try again."),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Download recording" }),
    ).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(await screen.findByTestId("step-hire")).toBeDefined();
    expect(finalizeBodies).toHaveLength(2);
    expect(finalizeBodies[1]).toMatchObject({
      input_mode: "typed",
      text: "I run a bakery",
    });
    expect(partUploads).toHaveLength(0);
  });
});

describe("onboarding brain dump — insufficient content", () => {
  async function recordAndSend() {
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
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Talk" }));
    await waitFor(() => expect(partUploads).toHaveLength(1));
    await userEvent.click(
      await screen.findByRole("button", { name: "Send recording" }),
    );
  }

  it("shows the recovery choices without advancing or losing the take", async () => {
    finalizeReturns({ status: "failed", error_code: "insufficient_content" });

    await recordAndSend();

    expect(
      await screen.findByText("We didn't catch enough of that."),
    ).toBeDefined();
    // Its own copy, not the transcription-failure screen's.
    expect(screen.queryByText("That didn't go through.")).toBeNull();
    expect(screen.getByRole("button", { name: "Record again" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Type instead" })).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Continue without it" }),
    ).toBeDefined();
    // No accidental advance to the personalized greeting...
    expect(screen.queryByTestId("step-hire")).toBeNull();
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBeNull();
    // ...and the local parts survive until the user picks a next move.
    expect(recordingStoreState.parts).toHaveLength(1);
  });

  it("offers a retry of the same take when the quality check was unavailable", async () => {
    finalizeReturns({
      status: "failed",
      error_code: "quality_check_unavailable",
    });

    await recordAndSend();

    expect(await screen.findByText("That didn't go through.")).toBeDefined();
    expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
    expect(screen.queryByText("We didn't catch enough of that.")).toBeNull();
    expect(recordingStoreState.parts).toHaveLength(1);
  });

  it("moves on from 'Continue without it' before the skip is recorded", async () => {
    const skip = holdSkipRecord();

    await recordAndSend();
    await screen.findByText("We didn't catch enough of that.");
    await userEvent.click(
      screen.getByRole("button", { name: "Continue without it" }),
    );

    expect(await screen.findByTestId("step-hire")).toBeDefined();
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
    await waitFor(() => expect(skip.skipBodies).toHaveLength(1));
    // The take stays for Back, and the skip is recorded as a take of its
    // own, so sending the recording later is not read as already done.
    expect(recordingStoreState.parts).toHaveLength(1);
    expect(skip.skipBodies[0]).not.toMatchObject({
      recording_id: recordingStoreState.parts[0].recordingId,
    });
    skip.release();
  });

  it("says why a take failed, and still offers the recording after skipping and going Back", async () => {
    finalizeReturns({ status: "failed", error_code: "storage_failed" });
    const objectURLs = [
      vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:dump"),
      vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined),
    ];
    const downloads: string[] = [];
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function recordDownload(this: HTMLAnchorElement) {
        downloads.push(this.download);
      });
    onTestFinished(() => {
      click.mockRestore();
      objectURLs.forEach((spy) => spy.mockRestore());
    });

    await recordAndSend();
    expect(await screen.findByText("That didn't go through.")).toBeDefined();
    expect(
      screen.getByText("We couldn't store your recording on our side."),
    ).toBeDefined();
    const recordingID = recordingStoreState.parts[0].recordingId;
    await userEvent.click(
      screen.getByRole("button", { name: "Continue without it" }),
    );
    await screen.findByTestId("step-hire");
    await userEvent.click(screen.getByRole("button", { name: "Back" }));

    expect(
      await screen.findByText("Pick up where you left off?"),
    ).toBeDefined();
    await userEvent.click(
      screen.getByRole("button", { name: "Download recording" }),
    );
    await waitFor(() =>
      expect(downloads).toEqual([`brain-dump-${recordingID}.webm`]),
    );
  });

  it("opens the typed composer from the recovery screen", async () => {
    finalizeReturns({ status: "failed", error_code: "no_usable_speech" });

    await recordAndSend();
    await screen.findByText("We didn't catch enough of that.");

    await userEvent.click(screen.getByRole("button", { name: "Type instead" }));

    expect(
      await screen.findByPlaceholderText(
        "What repeats every week? What would you hand off first?",
      ),
    ).toBeDefined();
  });
});

describe("onboarding brain dump — Otto might ask", () => {
  // The progress fixture's role.
  const ENGINEERING = BRAIN_DUMP_PROMPT_SETS.Engineering;

  beforeEach(() => {
    vi.mocked(posthog.capture).mockClear();
  });

  it("sets the role's starter ideas around the step, out of reach", async () => {
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);

    const ambient = screen.getByTestId("ambient-prompts");
    expect(ambient.getAttribute("aria-hidden")).toBe("true");
    expect(
      within(ambient).getByText(ENGINEERING.questions[0].text),
    ).toBeDefined();
    expect(within(ambient).getAllByText("For example")).toHaveLength(2);
    expect(ambient.querySelectorAll("button, a, [tabindex]")).toHaveLength(0);
  });

  it("asks one question at a time above the composer", async () => {
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Write" }));

    expect(
      await screen.findByText(ENGINEERING.questions[4].text),
    ).toBeDefined();
    await userEvent.click(
      screen.getByRole("button", { name: "Another question" }),
    );

    expect(
      await screen.findByText(ENGINEERING.questions[5].text),
    ).toBeDefined();
    expect(posthog.capture).toHaveBeenCalledWith("brain_dump_prompt_advanced", {
      via: "click",
      screen: "typing",
      prompt_index: 5,
      prompt_set: "Engineering",
    });
  });

  // The answer to WCAG 2.2.2: every idea that changes on its own can be
  // paused, from the keyboard or by clicking the question.
  function fakeRotationTimers() {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    onTestFinished(() => {
      vi.useRealTimers();
    });
  }

  async function waitFor20Seconds() {
    await act(async () => {
      vi.advanceTimersByTime(20_000);
    });
  }

  it("pauses the changing ideas from the keyboard", async () => {
    fakeRotationTimers();
    landOnPainPointsStep();
    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Write" }));
    await screen.findByText(ENGINEERING.questions[4].text);

    const toggle = screen.getByRole("button", {
      name: "Pause changing ideas",
    });
    toggle.focus();
    await userEvent.keyboard("{Enter}");
    expect(toggle.getAttribute("aria-pressed")).toBe("true");

    await waitFor20Seconds();
    expect(screen.getByText(ENGINEERING.questions[4].text)).toBeDefined();

    await userEvent.keyboard("{Enter}");
    expect(toggle.getAttribute("aria-pressed")).toBe("false");
    await waitFor20Seconds();
    expect(
      await screen.findByText(ENGINEERING.questions[5].text),
    ).toBeDefined();
  });

  it("pauses the changing ideas when the question is clicked", async () => {
    fakeRotationTimers();
    landOnPainPointsStep();
    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    // Below 1280px the live question at rest is the only pointer route.
    const question = screen
      .getAllByText(ENGINEERING.questions[4].text)
      .find((el) => !el.closest('[data-testid="ambient-prompts"]'));

    await userEvent.click(question!);
    await waitFor20Seconds();

    expect(
      screen
        .getAllByText(ENGINEERING.questions[4].text)
        .some((el) => !el.closest('[data-testid="ambient-prompts"]')),
    ).toBe(true);
    expect(
      screen
        .getByRole("button", { name: "Pause changing ideas" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
  });

  it("reports the prompt set and the prompts seen with a typed dump", async () => {
    finalizeReturns({ status: "completed" });
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Write" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Another question" }),
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: "Your brain dump" }),
      "Release notes every Friday",
    );
    await userEvent.click(screen.getByRole("button", { name: "I'm done" }));

    await waitFor(() =>
      expect(posthog.capture).toHaveBeenCalledWith("brain_dump_completed", {
        input_mode: "typed",
        chars: "Release notes every Friday".length,
        prompt_set: "Engineering",
        prompts_seen: 2,
      }),
    );
    expect(posthog.capture).toHaveBeenCalledWith("brain_dump_typed_fallback", {
      reason: "chose_to_type",
      prompt_set: "Engineering",
    });
  });

  it("reports the prompt set and the prompts seen with a skip", async () => {
    recordBrainDumpTraffic();
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Write" }));
    await screen.findByText(ENGINEERING.questions[4].text);
    await userEvent.click(screen.getByRole("button", { name: "Skip for now" }));

    expect(posthog.capture).toHaveBeenCalledWith("brain_dump_skipped", {
      prompt_set: "Engineering",
      prompts_seen: 1,
    });
    expect(await screen.findByTestId("step-hire")).toBeDefined();
  });

  it("sends the prompt set with the start of a recording", async () => {
    landOnPainPointsStep();

    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Talk" }));

    expect(
      await screen.findByRole("button", { name: "Send recording" }),
    ).toBeDefined();
    expect(posthog.capture).toHaveBeenCalledWith("brain_dump_started", {
      prompt_set: "Engineering",
    });
    expect(screen.getByText(ENGINEERING.questions[4].text)).toBeDefined();
  });
});

class FakePermissionStatus extends EventTarget {
  constructor(public state: PermissionState) {
    super();
  }

  change(state: PermissionState) {
    this.state = state;
    this.dispatchEvent(new Event("change"));
  }
}

describe("onboarding brain dump — a refused microphone", () => {
  const BLOCKED =
    "Your browser blocked the microphone, so write here instead. To talk, allow the microphone in your browser's site settings (the lock or microphone icon by the address bar), then try the microphone again.";
  const DISMISSED =
    "The microphone prompt was closed, so write here instead. To talk, try the microphone again.";
  const REFUSED = new DOMException("denied", "NotAllowedError");

  function captured(event: string) {
    return vi
      .mocked(posthog.capture)
      .mock.calls.filter(([name]) => name === event);
  }

  function capturedNames() {
    return vi.mocked(posthog.capture).mock.calls.map(([name]) => name);
  }

  function stubMicrophonePermission(state: PermissionState) {
    const status = new FakePermissionStatus(state);
    Object.defineProperty(navigator, "permissions", {
      configurable: true,
      value: { query: vi.fn().mockResolvedValue(status) },
    });
    onTestFinished(() => {
      Reflect.deleteProperty(navigator, "permissions");
    });
    return status;
  }

  async function pressTalkAndGetRefused() {
    landOnPainPointsStep();
    render(<OnboardingPage />);
    await screen.findByText(DUMP_HEADLINE);
    await userEvent.click(screen.getByRole("button", { name: "Talk" }));
    return screen.findByRole("alert");
  }

  beforeEach(() => {
    vi.mocked(posthog.capture).mockClear();
  });

  it("says why Talk did not start and how to allow the microphone", async () => {
    stubMicrophonePermission("denied");
    getUserMedia.mockRejectedValue(REFUSED);

    const notice = await pressTalkAndGetRefused();

    expect(notice.textContent).toBe(BLOCKED);
    expect(screen.getByText(TYPING_HEADLINE)).toBeDefined();
    expect(
      screen.getByRole("button", { name: "record instead" }),
    ).toBeDefined();
    expect(captured("brain_dump_permission_denied")).toHaveLength(1);
    expect(captured("brain_dump_typed_fallback")).toEqual([
      [
        "brain_dump_typed_fallback",
        { reason: "permission_denied", prompt_set: "Engineering" },
      ],
    ]);
  });

  it("tells a closed prompt apart from a blocked microphone", async () => {
    stubMicrophonePermission("prompt");
    getUserMedia.mockRejectedValue(REFUSED);

    const notice = await pressTalkAndGetRefused();

    expect(notice.textContent).toBe(DISMISSED);
  });

  it("starts recording when the retry is allowed", async () => {
    stubMicrophonePermission("prompt");
    getUserMedia
      .mockRejectedValueOnce(REFUSED)
      .mockResolvedValue({ getTracks: () => [] });
    await pressTalkAndGetRefused();

    await userEvent.click(
      screen.getByRole("button", { name: "try the microphone again" }),
    );

    expect(
      await screen.findByRole("button", { name: "Send recording" }),
    ).toBeDefined();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(captured("brain_dump_permission_retried")).toEqual([
      [
        "brain_dump_permission_retried",
        {
          result: "granted",
          permission_state: "prompt",
          via: "retry_link",
        },
      ],
    ]);
    const names = capturedNames();
    expect(names.indexOf("brain_dump_permission_retried")).toBeLessThan(
      names.indexOf("brain_dump_started"),
    );
  });

  it("keeps the notice and counts one fallback when refused again", async () => {
    stubMicrophonePermission("denied");
    getUserMedia.mockRejectedValue(REFUSED);
    await pressTalkAndGetRefused();

    await userEvent.click(
      screen.getByRole("button", { name: "record instead" }),
    );

    await waitFor(() =>
      expect(captured("brain_dump_permission_denied")).toHaveLength(2),
    );
    expect(captured("brain_dump_permission_retried")).toEqual([
      [
        "brain_dump_permission_retried",
        {
          result: "denied",
          permission_state: "denied",
          via: "record_instead",
        },
      ],
    ]);
    expect(captured("brain_dump_typed_fallback")).toHaveLength(1);
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(screen.getByRole("alert").textContent).toBe(BLOCKED);
    expect(screen.getByText(TYPING_HEADLINE)).toBeDefined();
  });

  it("keeps the notice up while the retry waits for an answer", async () => {
    stubMicrophonePermission("denied");
    getUserMedia.mockRejectedValueOnce(REFUSED);
    getUserMedia.mockReturnValueOnce(new Promise(() => {}));
    await pressTalkAndGetRefused();

    await userEvent.click(
      screen.getByRole("button", { name: "try the microphone again" }),
    );

    expect(getUserMedia).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("alert").textContent).toBe(BLOCKED);
  });

  it("says so when the retry finds the microphone busy", async () => {
    stubMicrophonePermission("denied");
    getUserMedia.mockRejectedValueOnce(REFUSED);
    getUserMedia.mockRejectedValueOnce(
      new DOMException("", "NotReadableError"),
    );
    await pressTalkAndGetRefused();

    await userEvent.click(
      screen.getByRole("button", { name: "try the microphone again" }),
    );

    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toContain(
        "We couldn't open your microphone, so write here instead.",
      ),
    );
    expect(screen.getByText(TYPING_HEADLINE)).toBeDefined();
  });

  it("clears the notice once the site is allowed from the address bar", async () => {
    const permission = stubMicrophonePermission("denied");
    getUserMedia.mockRejectedValue(REFUSED);
    await pressTalkAndGetRefused();

    act(() => {
      permission.change("granted");
    });

    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(captured("brain_dump_permission_retried")).toEqual([
      [
        "brain_dump_permission_retried",
        {
          result: "granted",
          permission_state: "granted",
          via: "permission_change",
        },
      ],
    ]);
    // "record instead" is the ordinary way back to Talk again.
    await userEvent.click(
      screen.getByRole("button", { name: "record instead" }),
    );
    expect(await screen.findByRole("button", { name: "Talk" })).toBeDefined();
  });
});

describe("onboarding step map integrity", () => {
  it("keeps personalization and hiring before the optional payment step", () => {
    expect(PAYWALL_LAST_STEPS).toEqual({
      team: 1,
      autopilot: 2,
      role: 3,
      painPoints: 4,
      hire: 5,
      subscription: 6,
      preparing: 7,
    });
    expect(NO_PAYWALL_STEPS).toEqual({
      team: 1,
      autopilot: 2,
      role: 3,
      painPoints: 4,
      hire: 5,
      preparing: 6,
    });
  });
});

function finalizeTextMaxLength() {
  const limit =
    openapi.components.schemas.FinalizeRequest.properties.text.anyOf.find(
      (branch) => branch.type === "string",
    )?.maxLength;
  expect(limit).toBeTypeOf("number");
  return limit!;
}

import { webcrypto } from "node:crypto";
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { PAYWALL_LAST_STEPS, useOnboardingWizardStore } from "../store";
import { useWizardProgress } from "../useWizardProgress";
import { landDraftSave, type StoredDraft } from "./draft-save-rule";
import { makeProgress as makeBaseProgress } from "./progress-fixture";

const { getState, patchState } = vi.hoisted(() => ({
  getState: vi.fn(),
  patchState: vi.fn(),
}));
vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1OnboardingState: getState,
  patchV1UpdateOnboardingState: patchState,
}));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

function makeProgress(overrides: Parameters<typeof makeBaseProgress>[0] = {}) {
  const draft = makeBaseProgress(overrides);
  return {
    ...draft,
    completedSteps: [
      "team",
      "autopilot",
      ...draft.completedSteps,
    ] as typeof draft.completedSteps,
  };
}

interface Save {
  wizardProgress: unknown;
  wizardRevision: number;
  wizardUserId: string;
  wizardWriterId?: string;
  wizardWriteSeq?: number;
}

function mountWizard() {
  return renderHook(
    ({ steps }) => useWizardProgress({ userID: "u1", ready: true, steps }),
    { initialProps: { steps: PAYWALL_LAST_STEPS } },
  );
}

async function settle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(250);
  });
}

function savesSent() {
  return patchState.mock.calls.map(([save]) => save as Save);
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.resetAllMocks();
  localStorage.clear();
  useOnboardingWizardStore.getState().reset();
  getState.mockResolvedValue({
    status: 200,
    data: {
      userId: "u1",
      completedSteps: [],
      wizardProgress: makeProgress(),
      wizardRevision: 4,
    },
  });
  patchState.mockImplementation(({ wizardRevision, wizardUserId }) =>
    Promise.resolve({
      status: 200,
      data: { userId: wizardUserId, wizardRevision: wizardRevision + 1 },
    }),
  );
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

it("names every save with one writer for the page and numbers the attempts 1, 2, 3", async () => {
  mountWizard();
  await settle();

  act(() => useOnboardingWizardStore.getState().setRole("Founder"));
  await settle();
  act(() => useOnboardingWizardStore.getState().setOtherRole("Solo"));
  await settle();
  act(() => useOnboardingWizardStore.getState().setOtherPainPoint("Slow"));
  await settle();

  const saves = savesSent();
  expect(saves.map((save) => save.wizardWriteSeq)).toEqual([1, 2, 3]);
  expect(saves[0].wizardWriterId).toEqual(expect.any(String));
  expect(new Set(saves.map((save) => save.wizardWriterId)).size).toBe(1);
});

it("keeps its writer and its numbering when the wizard reads the account again", async () => {
  const { rerender } = mountWizard();
  await settle();
  act(() => useOnboardingWizardStore.getState().setRole("Founder"));
  await settle();

  rerender({ steps: { ...PAYWALL_LAST_STEPS, preparing: 99 } });
  await settle();
  expect(getState).toHaveBeenCalledTimes(2);
  act(() => useOnboardingWizardStore.getState().setOtherRole("Solo"));
  await settle();

  const [before, after] = savesSent();
  expect(savesSent()).toHaveLength(2);
  expect(after.wizardWriterId).toBe(before.wizardWriterId);
  expect(after.wizardWriteSeq).toBe(before.wizardWriteSeq! + 1);
});

it.each([
  ["replays the draft the dropped save carried", true],
  ["goes on with a further edit", false],
])(
  "is not refused when the wizard reads the account again and it %s",
  async (_name, replays) => {
    let stored: StoredDraft<unknown> = {
      progress: makeProgress(),
      revision: 4,
      writerId: null,
      writeSeq: null,
    };
    let hold = false;
    const late: { land: (() => void) | null } = { land: null };
    const answers: number[] = [];
    function land(save: Save) {
      const next = landDraftSave(stored, save);
      if (!next) {
        answers.push(409);
        return { status: 409, data: { detail: "Changed in another session" } };
      }
      stored = next;
      answers.push(200);
      return {
        status: 200,
        data: { userId: "u1", wizardRevision: stored.revision },
      };
    }
    getState.mockImplementation(async () => ({
      status: 200,
      data: {
        userId: "u1",
        completedSteps: [],
        wizardProgress: stored.progress,
        wizardRevision: stored.revision,
      },
    }));
    patchState.mockImplementation((save: Save) => {
      if (!hold) return Promise.resolve(land(save));
      return new Promise((resolve) => {
        late.land = () => resolve(land(save));
      });
    });
    const { result, rerender } = mountWizard();
    await settle();
    act(() => useOnboardingWizardStore.getState().setRole("Founder"));
    await settle();
    hold = true;
    act(() => useOnboardingWizardStore.getState().setOtherRole("Solo"));
    await settle();
    expect(late.land).not.toBeNull();
    if (!replays) localStorage.clear();

    rerender({ steps: { ...PAYWALL_LAST_STEPS, preparing: 99 } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(100);
    });
    hold = false;
    await act(async () => late.land!());
    if (!replays)
      act(() => useOnboardingWizardStore.getState().setOtherPainPoint("Slow"));
    await settle();

    expect(answers).toEqual([200, 200, 200]);
    expect(result.current.conflict).toBe(false);
    expect(result.current.error).toBeNull();
    expect(stored.revision).toBe(7);
  },
);

it("names its saves where crypto.randomUUID is missing, as on plain HTTP", async () => {
  vi.stubGlobal("crypto", {
    getRandomValues: webcrypto.getRandomValues.bind(webcrypto),
  });
  expect(typeof crypto.randomUUID).toBe("undefined");

  mountWizard();
  await settle();
  act(() => useOnboardingWizardStore.getState().setRole("Founder"));
  await settle();

  expect(savesSent()).toHaveLength(1);
  expect(savesSent()[0].wizardWriterId).toMatch(
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
  );
});

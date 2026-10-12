import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { readLocalProgress, writeLocalProgress } from "../progress";
import { PAYWALL_LAST_STEPS, useOnboardingWizardStore } from "../store";
import { useWizardProgress } from "../useWizardProgress";
import { makeProgress } from "./progress-fixture";

const { getState, patchState } = vi.hoisted(() => ({
  getState: vi.fn(),
  patchState: vi.fn(),
}));
vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1OnboardingState: getState,
  patchV1UpdateOnboardingState: patchState,
}));
let params = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  useSearchParams: () => params,
}));

function response(progress = makeProgress(), revision = 4, userID = "u1") {
  return {
    status: 200,
    data: {
      userId: userID,
      completedSteps: [],
      wizardProgress: progress,
      wizardRevision: revision,
    },
  };
}

function mountWizard(userID = "u1") {
  return renderHook(
    ({ userID }) =>
      useWizardProgress({
        userID,
        ready: true,
        steps: PAYWALL_LAST_STEPS,
      }),
    { initialProps: { userID } },
  );
}

async function settle() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(250);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.resetAllMocks();
  params = new URLSearchParams();
  localStorage.clear();
  useOnboardingWizardStore.getState().reset();
  getState.mockResolvedValue(response());
  patchState.mockImplementation(({ wizardRevision, wizardUserId }) =>
    Promise.resolve({
      status: 200,
      data: { userId: wizardUserId, wizardRevision: wizardRevision + 1 },
    }),
  );
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

it("restores a server draft without saving or depending on property order", async () => {
  const reordered = Object.fromEntries(
    Object.entries(makeProgress()).reverse(),
  );
  getState.mockResolvedValue({
    ...response(),
    data: { ...response().data, wizardProgress: reordered },
  });
  const { result } = mountWizard();
  await settle();
  expect(result.current.isReady).toBe(true);
  expect(patchState).not.toHaveBeenCalled();
  expect(readLocalProgress("u1")).toEqual({
    progress: makeProgress(),
    revision: 4,
    pending: false,
  });
});

it("does not save an empty wizard until its first edit", async () => {
  getState.mockResolvedValue({
    ...response(),
    data: { ...response().data, wizardProgress: null, wizardRevision: 0 },
  });
  mountWizard();
  await settle();
  expect(patchState).not.toHaveBeenCalled();
  act(() => useOnboardingWizardStore.getState().setRole("Founder"));
  await settle();
  expect(patchState).toHaveBeenCalledExactlyOnceWith(
    expect.objectContaining({
      wizardProgress: expect.objectContaining({ role: "Founder" }),
      wizardRevision: 0,
      wizardUserId: "u1",
    }),
    expect.anything(),
  );
});

it("saves the first edit after a clean restore with the loaded revision", async () => {
  mountWizard();
  await settle();
  act(() => useOnboardingWizardStore.getState().setOtherRole("Founder"));
  await settle();
  expect(patchState).toHaveBeenCalledExactlyOnceWith(
    expect.objectContaining({
      wizardProgress: makeProgress({ otherRole: "Founder" }),
      wizardRevision: 4,
    }),
    expect.anything(),
  );
});

it.each([false, true])(
  "replays pending changes when offline=%s",
  async (offline) => {
    const changed = makeProgress({ role: "Founder" });
    writeLocalProgress("u1", changed, true, 4);
    if (offline) getState.mockRejectedValue(new Error("Offline"));
    mountWizard();
    await settle();
    expect(patchState).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ wizardProgress: changed, wizardRevision: 4 }),
      expect.anything(),
    );
  },
);

it("acknowledges an uncertain save already present on the server without another write", async () => {
  writeLocalProgress("u1", makeProgress(), true, 3);
  mountWizard();
  await settle();
  expect(patchState).not.toHaveBeenCalled();
  expect(readLocalProgress("u1")).toEqual({
    progress: makeProgress(),
    revision: 4,
    pending: false,
  });
});

it("keeps a clean offline cache clean and saves subsequent edits", async () => {
  writeLocalProgress("u1", makeProgress(), false, 4);
  getState.mockRejectedValue(new Error("Offline"));
  const { result } = mountWizard();
  await settle();
  expect(patchState).not.toHaveBeenCalled();
  expect(result.current.error).not.toMatch(/retrying/i);
  act(() => useOnboardingWizardStore.getState().setRole("Founder"));
  await settle();
  expect(patchState).toHaveBeenCalledOnce();
});

it("persists a requested-step correction", async () => {
  params = new URLSearchParams("step=role");
  getState.mockResolvedValue(
    response(
      makeProgress({
        currentStep: "subscription",
        completedSteps: ["role", "painPoints"],
      }),
    ),
  );
  mountWizard();
  await settle();
  expect(patchState).toHaveBeenCalledExactlyOnceWith(
    expect.objectContaining({
      wizardProgress: makeProgress({
        currentStep: "role",
        completedSteps: ["role", "painPoints"],
      }),
      wizardRevision: 4,
    }),
    expect.anything(),
  );
});

it("does not acknowledge a different outbox written by another tab during restore", async () => {
  writeLocalProgress("u1", makeProgress(), true, 3);
  getState.mockImplementation(async () => {
    writeLocalProgress("u1", makeProgress({ role: "Newer tab" }), true, 4);
    return response();
  });
  mountWizard();
  await settle();
  expect(patchState).not.toHaveBeenCalled();
  expect(readLocalProgress("u1")).toEqual({
    progress: makeProgress({ role: "Newer tab" }),
    revision: 4,
    pending: true,
  });
});

it("retains stale-draft conflict handling and reloads without a redundant write", async () => {
  writeLocalProgress("u1", makeProgress({ role: "Stale" }), true, 3);
  const { result } = mountWizard();
  await settle();
  expect(result.current.conflict).toBe(true);
  expect(patchState).not.toHaveBeenCalled();
  act(() => result.current.retry());
  await settle();
  expect(result.current.isReady).toBe(true);
  expect(patchState).not.toHaveBeenCalled();
  expect(readLocalProgress("u1")?.pending).toBe(false);
});

it("rejects a restored draft belonging to another signed-in account", async () => {
  getState.mockResolvedValue(response(makeProgress(), 4, "u2"));
  const { result } = mountWizard();
  await settle();
  expect(result.current.isReady).toBe(false);
  expect(result.current.error).toMatch(/account changed/i);
  expect(readLocalProgress("u1")).toBeNull();
  expect(patchState).not.toHaveBeenCalled();
});

it("ignores a delayed restore after the account changes", async () => {
  let resolveFirst!: (value: ReturnType<typeof response>) => void;
  getState.mockReturnValueOnce(
    new Promise((resolve) => {
      resolveFirst = resolve;
    }),
  );
  getState.mockResolvedValueOnce(
    response(makeProgress({ role: "Other account" }), 6, "u2"),
  );
  const { rerender } = mountWizard();
  rerender({ userID: "u2" });
  await settle();
  await act(async () => {
    resolveFirst(response());
  });
  expect(useOnboardingWizardStore.getState()).toMatchObject({
    userID: "u2",
    role: "Other account",
  });
  expect(readLocalProgress("u1")).toBeNull();
  expect(patchState).not.toHaveBeenCalled();
});

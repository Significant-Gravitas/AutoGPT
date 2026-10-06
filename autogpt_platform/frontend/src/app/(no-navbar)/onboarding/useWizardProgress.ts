import { patchV1UpdateOnboardingState } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { clearLocalProgress, resumeStep, sameProgress } from "./progress";
import { loadWizardProgress } from "./loadWizardProgress";
import { createProgressSaver } from "./progress-saver";
import {
  snapshotProgress,
  type StepLayout,
  useOnboardingWizardStore,
} from "./store";

export function useWizardProgress({
  userID,
  ready,
  steps,
}: {
  userID: string | null;
  ready: boolean;
  steps: StepLayout;
}) {
  const router = useRouter();
  const params = useSearchParams();
  const currentUserID = useRef(userID);
  currentUserID.current = userID;
  const latest = useRef({ steps, params, router });
  latest.current = { steps, params, router };
  const layout = JSON.stringify(steps);
  const reloadLatest = useRef(false);
  const accountChanged = useRef(false);
  const [attempt, setAttempt] = useState(0);
  const [status, setStatus] = useState({
    userID,
    layout: "",
    loaded: false,
    loadError: false,
    conflict: false,
    error: null as string | null,
  });
  const saver = useRef<ReturnType<typeof createProgressSaver> | null>(null);

  useEffect(() => {
    if (!ready || !userID) return;
    let active = true;
    let unsubscribe: (() => void) | undefined;
    const abort = new AbortController();
    const { steps, params, router } = latest.current;
    const isCurrent = () => active && currentUserID.current === userID;
    useOnboardingWizardStore.getState().reset();
    setStatus({
      userID,
      layout,
      loaded: false,
      loadError: false,
      conflict: false,
      error: null,
    });

    function conflict() {
      if (isCurrent())
        setStatus({
          userID,
          layout,
          loaded: false,
          loadError: true,
          conflict: true,
          error:
            "Onboarding was updated in another session. Reload its latest progress to continue. This device's unsaved answers will be replaced.",
        });
    }

    async function initialize() {
      const result = await loadWizardProgress(
        userID!,
        reloadLatest.current,
        abort,
      );
      if (!isCurrent()) return;
      if (result.kind === "complete") {
        clearLocalProgress(userID!);
        router.replace("/copilot");
        return;
      }
      if (result.kind === "conflict") {
        conflict();
        return;
      }
      if (result.kind !== "ready") {
        accountChanged.current = result.kind === "accountChanged";
        setStatus({
          userID,
          layout,
          loaded: false,
          loadError: true,
          conflict: false,
          error: accountChanged.current
            ? "Your signed-in account changed. Reload the page to restore its progress."
            : "We couldn't restore your onboarding progress. Please retry.",
        });
        return;
      }
      const { progress, revision, offline, needsSync } = result;
      if (!offline) reloadLatest.current = false;
      const controller = createProgressSaver({
        userID: userID!,
        initialRevision: revision,
        onConflict: conflict,
        async save(wizardProgress, wizardRevision, signal) {
          if (!isCurrent()) throw new Error("Your account changed.");
          const result = await patchV1UpdateOnboardingState(
            { wizardProgress, wizardRevision, wizardUserId: userID! },
            { signal },
          );
          if (result.status !== 200) {
            throw Object.assign(new Error("Could not save onboarding."), {
              status: result.status,
            });
          }
          if (result.data.userId !== userID) {
            accountChanged.current = true;
            throw Object.assign(new Error("Your signed-in account changed."), {
              status: 409,
            });
          }
          return result.data.wizardRevision ?? wizardRevision + 1;
        },
        onError(error) {
          if (isCurrent()) setStatus((state) => ({ ...state, error }));
        },
      });
      saver.current = controller;
      const currentStep = resumeStep({
        progress,
        steps,
        requestedStep: params.get("step"),
      });
      const { currentStep: savedStep, version, ...fields } = progress ?? {};
      void savedStep;
      void version;
      useOnboardingWizardStore.setState({
        ...fields,
        userID,
        steps,
        currentStep,
        flushProgress: controller.flush,
      });
      setStatus({
        userID,
        layout,
        loaded: true,
        loadError: false,
        conflict: false,
        error: offline
          ? "Your saved progress is available on this device. Changes will sync when the connection returns."
          : null,
      });
      let previous = snapshotProgress();
      if (needsSync || (progress && !sameProgress(previous, progress)))
        controller.enqueue(previous);
      unsubscribe = useOnboardingWizardStore.subscribe((state) => {
        if (!isCurrent() || state.userID !== userID) return;
        const snapshot = snapshotProgress();
        if (sameProgress(snapshot, previous)) return;
        previous = snapshot;
        controller.enqueue(snapshot);
      });
    }
    void initialize();
    return () => {
      active = false;
      abort.abort();
      unsubscribe?.();
      saver.current?.dispose();
      saver.current = null;
      if (currentUserID.current !== userID)
        useOnboardingWizardStore.getState().reset();
    };
  }, [userID, ready, layout, attempt]);

  function retry() {
    if (accountChanged.current) {
      window.location.reload();
      return;
    }
    if (status.loadError) {
      reloadLatest.current = status.conflict;
      setAttempt((value) => value + 1);
    } else void saver.current?.flush().catch(() => undefined);
  }

  function finish() {
    saver.current?.dispose();
    if (userID) clearLocalProgress(userID);
  }

  return {
    isReady:
      status.loaded && status.userID === userID && status.layout === layout,
    error: status.userID === userID ? status.error : null,
    conflict: status.userID === userID && status.conflict,
    retry,
    finish,
  };
}

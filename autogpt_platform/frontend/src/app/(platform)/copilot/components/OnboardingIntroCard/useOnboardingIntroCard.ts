import { peekFirstRunTourSeen } from "@/app/(platform)/components/FirstRunTour/helpers";
import { useGetBrainDumpIntro } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import {
  clearWelcomePending,
  peekGreetingDone,
  peekIntroPath,
  peekWelcomePending,
  setGreetingDone,
  setIntroAwaitingFollowup,
  setMicGlow,
  setPendingLaterDump,
  setWelcomePending,
  takeIntroPath,
} from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useState } from "react";

// While the pipeline is still writing the greeting the intro endpoint
// answers `greeting_pending` — poll at this cadence until the real
// greeting lands (or the pipeline terminally resolves).
const PENDING_POLL_MS = 1500;
// A pipeline killed between transcription and completion leaves the dump
// in a non-terminal status forever — without a ceiling this poll would
// spin for the rest of the session and the composer would never appear.
const PENDING_GIVE_UP_MS = 120_000;

export function useOnboardingIntroCard() {
  const { user } = useAuth();
  const userId = user?.id ?? null;
  // localStorage answers first so a returning user never flashes the
  // greeting; only when it has no answer do we ask the server. The flag
  // is keyed to the user id, so a fresh account on the same browser
  // starts clean.
  const [isDone, setIsDone] = useState(() => peekGreetingDone(userId));

  // A user fresh out of onboarding gets the guided first run over the real
  // sidebar (FirstRunTour). The greeting loads behind it rather than after
  // it: the walkthrough ends by pointing at the greeting's first tasks.
  // Seeded from sessionStorage so a refresh mid-walkthrough brings it back
  // rather than skipping it.
  const [isFirstRunPending, setIsFirstRunPending] = useState(
    () => peekWelcomePending() || Boolean(peekIntroPath()),
  );

  // The server render can't see localStorage or the intro answer, so it
  // always emits the held hero; the first client frame must match it or
  // hydration flashes the composer before the greeting page arrives.
  // Everyone gets one held frame, then the real state takes over.
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => setIsMounted(true), []);

  // Give-up switch for a greeting pipeline that never terminates; once
  // flipped the hero releases the composer instead of holding forever.
  const [gaveUpWaiting, setGaveUpWaiting] = useState(false);

  useEffect(() => {
    // The user record can arrive after mount — re-check once it does.
    if (peekGreetingDone(userId)) setIsDone(true);
  }, [userId]);

  useEffect(() => {
    // This browser already saw this user finish or skip the walkthrough —
    // never reopen it, even if the pending flag somehow survived.
    if (isFirstRunPending && peekFirstRunTourSeen(userId)) {
      clearWelcomePending();
      setIsFirstRunPending(false);
    }
  }, [isFirstRunPending, userId]);

  useEffect(() => {
    const path = takeIntroPath();
    if (!path) return;
    if (path === "B") {
      setMicGlow();
      // Path B's intro asks for the dump from the composer instead, so the
      // first voice message there is the answer to that invitation.
      setPendingLaterDump();
    }
    // Measures whether the greeting actually started a conversation —
    // consumed by the first real message afterwards.
    setIntroAwaitingFollowup();
    trackBrainDump("intro_path", { path });
    if (!peekFirstRunTourSeen(userId)) {
      setWelcomePending();
      setIsFirstRunPending(true);
    }
  }, []);

  const { data, isError } = useGetBrainDumpIntro({
    query: {
      enabled: !isDone,
      staleTime: Infinity,
      gcTime: Infinity,
      refetchInterval: (query) => {
        if (gaveUpWaiting) return false;
        const latest = query.state.data;
        if (!latest || latest.status !== 200) return false;
        const body = latest.data;
        if (body.greeting_done) return false;
        if (body.greeting_pending) return PENDING_POLL_MS;
        return false;
      },
    },
  });

  const intro = data?.status === 200 ? data.data : null;
  // "Answered" rather than "not loading": isLoading is false on the
  // server render and on the first client frames before the fetch
  // subscribes, which flashed the regular hero + composer for a beat
  // before the greeting page took over on refresh. A non-200 or a
  // failed request counts as an answer so the composer is never held
  // hostage by an unavailable endpoint.
  const hasIntroAnswer = data !== undefined || isError;
  const serverSaysDone = Boolean(intro?.greeting_done);
  const isPendingPerServer = Boolean(
    intro && !intro.greeting_done && intro.greeting_pending,
  );
  const isPendingGeneration = isPendingPerServer && !gaveUpWaiting;

  useEffect(() => {
    if (!isPendingPerServer) return;
    const timer = setTimeout(() => setGaveUpWaiting(true), PENDING_GIVE_UP_MS);
    return () => clearTimeout(timer);
  }, [isPendingPerServer]);

  useEffect(() => {
    // The server already saw the first message (possibly from another
    // device) — cache that locally so we never ask again.
    if (serverSaysDone && userId) {
      setGreetingDone(userId);
      setIsDone(true);
    }
  }, [serverSaysDone, userId]);

  // The first run is over: the walkthrough ended, or never had to open.
  function finishFirstRun({ wasShown }: { wasShown: boolean }) {
    if (wasShown) trackBrainDump("welcome_dialog_closed", {});
    clearWelcomePending();
    setIsFirstRunPending(false);
  }

  // `serverSaysDone` as well as `isDone`: the effect that caches the verdict
  // runs after render, so a payload that still carries a greeting would
  // otherwise flash the card for one paint before it takes effect.
  const isVisible =
    isMounted &&
    !isDone &&
    !serverSaysDone &&
    hasIntroAnswer &&
    Boolean(intro?.greeting);

  return {
    isVisible,
    // The greeting is on its way (no server answer yet, or the pipeline
    // still writing): the page shows the orb alone, centered, and
    // holds the composer until the greeting takes over. The orb then
    // travels into the heading rather than being replaced by it, so this
    // must go false in the same commit `isVisible` goes true.
    //
    // The pre-mount frame counts: the server render cannot know any of
    // this, and letting the composer through on that one frame flashed it
    // before the greeting page arrived. The loader fades in on a delay, so
    // the first frame does not flash the composer.
    isAwaitingGreeting:
      !isMounted || (!isDone && (!hasIntroAnswer || isPendingGeneration)),
    // Only the greeting itself reads top-down like a letter. The loader
    // stays centered — that is the distance the orb travels when the
    // greeting lands.
    anchorTop: isVisible,
    // Independent of the greeting: whether someone has had the walkthrough
    // has nothing to do with whether they have sent their first message.
    isFirstRunPending,
    finishFirstRun,
    greeting: intro?.greeting ?? "",
    transcript: typeof intro?.transcript === "string" ? intro.transcript : "",
    path: intro?.path ?? "B",
  };
}

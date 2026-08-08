import { useGetBrainDumpIntro } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import {
  clearWelcomePending,
  peekCapabilityCardsSeen,
  peekGreetingDone,
  peekWelcomePending,
  setCapabilityCardsSeen,
  setGreetingDone,
  setIntroAwaitingFollowup,
  setMicGlow,
  setPendingLaterDump,
  setWelcomePending,
  takeIntroPath,
} from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useState } from "react";

// While the pipeline is still writing the greeting the intro endpoint
// answers with an empty Path A — poll at this cadence until the real
// greeting lands (or the pipeline terminally resolves).
const PENDING_POLL_MS = 1500;
// A pipeline killed between transcription and completion leaves the dump
// in a non-terminal status forever — without a ceiling this poll would
// spin for the rest of the session and the composer would never appear.
const PENDING_GIVE_UP_MS = 120_000;

export function useOnboardingIntroCard() {
  const { user } = useAuth();
  const userId = user?.id ?? null;
  // The endpoint 404s with the flag off, so without this every copilot
  // visit for every user pays for a request that cannot succeed.
  const { enabled: isBrainDumpEnabled, ready: isFlagReady } = useFlagStatus(
    Flag.ONBOARDING_BRAIN_DUMP,
  );

  // localStorage answers first so a returning user never flashes the
  // greeting; only when it has no answer do we ask the server. The flag
  // is keyed to the user id, so a fresh account on the same browser
  // starts clean.
  const [isDone, setIsDone] = useState(() => peekGreetingDone(userId));

  // A user fresh out of onboarding gets the full-screen welcome overlay
  // first. The greeting is not even fetched until they close it — the
  // reveal animation must start on a page they are actually looking at.
  // Seeded from sessionStorage so a refresh keeps the overlay up rather
  // than skipping it.
  const [isWelcomeOpen, setIsWelcomeOpen] = useState(peekWelcomePending);

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
    // This browser already saw this user finish or skip the cards —
    // never reshow, even if the pending flag somehow survived.
    if (isWelcomeOpen && peekCapabilityCardsSeen(userId)) {
      clearWelcomePending();
      setIsWelcomeOpen(false);
    }
  }, [isWelcomeOpen, userId]);

  useEffect(() => {
    // The handoff keys are written by the wizard while the flag is on, so
    // they can outlive it: wait for LaunchDarkly, then either run the
    // handoff or drop it. Consuming it with the flag off would put the
    // overlay and its 404-ing polls in front of a rolled-back user.
    if (!isFlagReady) return;
    if (!isBrainDumpEnabled) {
      takeIntroPath();
      clearWelcomePending();
      setIsWelcomeOpen(false);
      return;
    }
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
    setWelcomePending();
    setIsWelcomeOpen(true);
  }, [isFlagReady, isBrainDumpEnabled]);

  const { data, isError } = useGetBrainDumpIntro({
    query: {
      enabled: Boolean(isBrainDumpEnabled) && !isDone && !isWelcomeOpen,
      staleTime: Infinity,
      gcTime: Infinity,
      refetchInterval: (query) => {
        if (gaveUpWaiting) return false;
        const latest = query.state.data;
        if (!latest || latest.status !== 200) return false;
        const body = latest.data;
        // Empty Path A greeting = pipeline still generating; keep asking.
        if (!body.greeting_done && body.path === "A" && !body.greeting) {
          return PENDING_POLL_MS;
        }
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
  // hostage by an unavailable endpoint — and so does a flag we know is
  // off, since that query is disabled and will never answer at all. Wait
  // for LaunchDarkly to actually say so: treating "not answered yet" as
  // off is what flashes the composer before the greeting page arrives.
  const hasIntroAnswer =
    (isFlagReady && !isBrainDumpEnabled) || data !== undefined || isError;
  const serverSaysDone = Boolean(intro?.greeting_done);
  const isPendingPerServer = Boolean(
    intro && !intro.greeting_done && intro.path === "A" && !intro.greeting,
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

  function closeWelcome() {
    trackBrainDump("welcome_dialog_closed", {});
    clearWelcomePending();
    setCapabilityCardsSeen(userId);
    setIsWelcomeOpen(false);
  }

  // The whole greeting flow reads top-down like a letter, so it anchors
  // to the top from its first visible frame — flipping the container
  // from centered to top only when the greeting arrived made the "Hey"
  // heading visibly jump. `isWelcomeOpen` needs no flag check (only the
  // gated handoff ever sets it) and is seeded synchronously, so the
  // fresh-out-of-onboarding user is anchored before LaunchDarkly answers.
  const isGreetingFlow =
    isWelcomeOpen ||
    (Boolean(isBrainDumpEnabled) &&
      (!hasIntroAnswer || isPendingGeneration || Boolean(intro?.greeting)));

  // Holding the composer is only ever right while this flow is in play:
  // the overlay is up (seeded synchronously from the handoff) or the flag
  // already reads on. Holding on "LaunchDarkly has not answered" alone
  // hid the composer on every flag-off /copilot load until it did.
  const isGreetingExpected = isWelcomeOpen || Boolean(isBrainDumpEnabled);

  return {
    // The page renders the regular hero behind the welcome modal and
    // while the greeting is generating; it swaps to the greeting the
    // moment the real one arrives.
    isVisible:
      isMounted &&
      !isDone &&
      !isWelcomeOpen &&
      hasIntroAnswer &&
      Boolean(intro?.greeting),
    // The greeting is on its way (modal up, no server answer yet, or the
    // pipeline still writing) — the hero shows the orb and holds the
    // composer back until the greeting page takes over.
    isAwaitingGreeting:
      !isMounted ||
      (!isDone &&
        isGreetingExpected &&
        (isWelcomeOpen || !hasIntroAnswer || isPendingGeneration)),
    anchorTop: isMounted && !isDone && isGreetingFlow,
    isWelcomeOpen: !isDone && isWelcomeOpen,
    closeWelcome,
    greeting: intro?.greeting ?? "",
    prompts: intro?.prompts ?? [],
    transcript: typeof intro?.transcript === "string" ? intro.transcript : "",
    path: intro?.path ?? "B",
  };
}

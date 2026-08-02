import { useGetBrainDumpIntro } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
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
  setWelcomePending,
  takeIntroPath,
} from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useState } from "react";

// While the pipeline is still writing the greeting the intro endpoint
// answers with an empty Path A — poll at this cadence until the real
// greeting lands (or the pipeline terminally resolves).
const PENDING_POLL_MS = 1500;

export function useOnboardingIntroCard() {
  const { user } = useAuth();
  const userId = user?.id ?? null;
  // The endpoint 404s with the flag off, so without this every copilot
  // visit for every user pays for a request that cannot succeed.
  const isBrainDumpEnabled = useGetFlag(Flag.ONBOARDING_BRAIN_DUMP);

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
    const path = takeIntroPath();
    if (!path) return;
    if (path === "B") setMicGlow();
    // Measures whether the greeting actually started a conversation —
    // consumed by the first real message afterwards.
    setIntroAwaitingFollowup();
    trackBrainDump("intro_path", { path });
    setWelcomePending();
    setIsWelcomeOpen(true);
  }, []);

  const { data, isError } = useGetBrainDumpIntro({
    query: {
      enabled: Boolean(isBrainDumpEnabled) && !isDone && !isWelcomeOpen,
      staleTime: Infinity,
      gcTime: Infinity,
      refetchInterval: (query) => {
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
  // hostage by an unavailable endpoint.
  const hasIntroAnswer = data !== undefined || isError;
  const serverSaysDone = Boolean(intro?.greeting_done);
  const isPendingGeneration = Boolean(
    intro && !intro.greeting_done && intro.path === "A" && !intro.greeting,
  );

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
      (!isDone && (isWelcomeOpen || !hasIntroAnswer || isPendingGeneration)),
    isWelcomeOpen: !isDone && isWelcomeOpen,
    closeWelcome,
    greeting: intro?.greeting ?? "",
    prompts: intro?.prompts ?? [],
    transcript: typeof intro?.transcript === "string" ? intro.transcript : "",
    path: intro?.path ?? "B",
  };
}

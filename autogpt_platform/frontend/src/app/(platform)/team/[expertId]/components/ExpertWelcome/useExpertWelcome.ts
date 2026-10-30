import {
  draftExpertSoul,
  getExpert,
  getGetExpertQueryKey,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { toast } from "@/components/molecules/Toast/use-toast";
import { waitForRaise } from "@/services/experts/raise-expert";
import { raiseIntroQueryOptions } from "@/services/experts/raise-intro";
import {
  useQuery,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import { useState } from "react";
import {
  isEveryQuestionAnswered,
  markWelcomeDone,
  toSoulAnswers,
  type WelcomeAnswers,
} from "./helpers";

const SOUL_POLL_MS = 1_500;
const SOUL_POLL_LIMIT = 20;

type Phase = "asking" | "answered" | "skipped";

export function useExpertWelcome(expert: Expert) {
  const queryClient = useQueryClient();
  const introQuery = useQuery(
    raiseIntroQueryOptions(expert.name, expert.job_title ?? ""),
  );
  const intro = introQuery.data ?? null;
  // An opening that was prefetched while the dialog was open is part of the
  // first paint and does not animate in; one that arrives later does.
  const [hadIntroAtMount] = useState(() => intro !== null);
  const [answers, setAnswers] = useState<WelcomeAnswers>({});
  const [phase, setPhase] = useState<Phase>("asking");

  function answer(index: number, value: string) {
    if (!intro || phase !== "asking") return;
    const next = { ...answers, [index]: value };
    setAnswers(next);
    if (!isEveryQuestionAnswered(intro.questions, next)) return;
    setPhase("answered");
    markWelcomeDone(expert.id);
    void saveSoul(queryClient, expert.id, {
      intro: intro.intro,
      answers: toSoulAnswers(intro.questions, next),
    });
  }

  function skip() {
    setPhase("skipped");
    markWelcomeDone(expert.id);
  }

  return {
    intro,
    answers,
    // The opening never fails on the server (a template stands in for the
    // model), so an error here is the network: let the user go straight on.
    phase: introQuery.isError && !intro ? ("skipped" as const) : phase,
    animateIntro: !hadIntroAtMount,
    answer,
    skip,
  };
}

async function saveSoul(
  queryClient: QueryClient,
  expertId: string,
  body: Parameters<typeof draftExpertSoul>[1],
) {
  try {
    await waitForRaise(expertId);
    const queued = await draftExpertSoul(expertId, body);
    if (queued.status !== 202) throw new Error("Draft not queued");
    await refreshWhenSoulChanges(
      queryClient,
      expertId,
      queued.data.expected_identity,
    );
  } catch {
    toast({
      title: "Couldn't save your answers",
      description: "You can tell your expert the same thing in chat.",
      variant: "destructive",
    });
  }
}

// The soul is written in the background; the page picks it up as soon as it
// lands rather than on the next manual refresh.
async function refreshWhenSoulChanges(
  queryClient: QueryClient,
  expertId: string,
  previousIdentity: string,
) {
  for (let attempt = 0; attempt < SOUL_POLL_LIMIT; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, SOUL_POLL_MS));
    const response = await getExpert(expertId);
    if (response.status !== 200) return;
    if (response.data.identity !== previousIdentity) {
      queryClient.setQueryData(getGetExpertQueryKey(expertId), response);
      return;
    }
  }
}

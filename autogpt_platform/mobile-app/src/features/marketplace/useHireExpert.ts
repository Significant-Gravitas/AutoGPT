import {
  type Expert,
  type ExpertTemplate,
  expertKeys,
  hireExpert,
} from "@/lib/api/experts";
import { isSessionGone, queryClient, sessionGeneration, signOut } from "@/lib/session";

import { hireErrorMessage, withHiredExpert } from "./helpers";
import {
  claimHire,
  hireStateFor,
  hiresSince,
  readHireState,
  recordHire,
  releaseHire,
  useHireStore,
} from "./hireStore";

interface Args {
  onHired: (expert: Expert, template: ExpertTemplate) => void;
  onError: (message: string, template: ExpertTemplate) => void;
}

export function useHireExpert({ onHired, onError }: Args) {
  const store = hireStateFor(useHireStore(), sessionGeneration());

  async function hire(template: ExpertTemplate) {
    const generation = sessionGeneration();
    if (!claimHire(template.id, generation)) return;
    try {
      const { expert } = await hireExpert(template.id);
      // Signed out meanwhile: the old account's expert stays out of the cache.
      if (generation !== sessionGeneration()) return;
      // Kept apart from the team list, so the hire shows even if that list
      // never loads; written into the list only when it has loaded.
      recordHire(template.id, expert, generation);
      queryClient.setQueryData<Expert[]>(expertKeys.list, (old) =>
        old ? withHiredExpert(old, expert) : old,
      );
      // A list request that started before the hire mustn't be the load that
      // decides it (`hiresSince`). Invalidating alone reuses a first load still
      // in flight, so cancel whatever is running, then load afresh.
      void queryClient
        .cancelQueries({ queryKey: expertKeys.list })
        .then(() => queryClient.invalidateQueries({ queryKey: expertKeys.list }));
      onHired(expert, template);
    } catch (caught) {
      // A refusal for an account already signed out must not end the next one.
      if (generation !== sessionGeneration()) return;
      if (isSessionGone(caught)) {
        void signOut();
        return;
      }
      onError(hireErrorMessage(caught, template.name), template);
    } finally {
      releaseHire(template.id, generation);
    }
  }

  function isHiring(templateID: string) {
    return store.inFlight.has(templateID);
  }

  /**
   * The expert this sign-in just hired from the template, if any. Read when
   * called, so the hire toast's Chat button sees the hire it announces.
   */
  function justHired(templateID: string) {
    return hireStateFor(readHireState(), sessionGeneration()).hired.get(templateID) ?? null;
  }

  /** Hires the team list, last loaded at `listUpdatedAt`, can't have yet. */
  function newerHires(listUpdatedAt: number) {
    return hiresSince(store, listUpdatedAt);
  }

  return { hire, isHiring, justHired, newerHires };
}

import {
  type Expert,
  type ExpertTemplate,
  expertKeys,
  hireExpert,
} from "@/lib/api/experts";
import { isSessionGone, queryClient, sessionGeneration, signOut } from "@/lib/session";

import { hireErrorMessage, withHiredExpert } from "./helpers";
import { claimHire, releaseHire, useHiresInFlight } from "./hiresInFlight";

interface Args {
  onHired: (expert: Expert, template: ExpertTemplate) => void;
  onError: (message: string, template: ExpertTemplate) => void;
}

export function useHireExpert({ onHired, onError }: Args) {
  const hiring = useHiresInFlight();

  async function hire(template: ExpertTemplate) {
    if (!claimHire(template.id)) return;
    const generation = sessionGeneration();
    try {
      const { expert } = await hireExpert(template.id);
      // Signed out meanwhile: the old account's expert stays out of the cache.
      if (generation !== sessionGeneration()) return;
      // Only into a list that loaded: a team of one would hide a failed load.
      queryClient.setQueryData<Expert[]>(expertKeys.list, (old) =>
        old ? withHiredExpert(old, expert) : old,
      );
      void queryClient.invalidateQueries({ queryKey: expertKeys.list });
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
      releaseHire(template.id);
    }
  }

  function isHiring(templateID: string) {
    return hiring.has(templateID);
  }

  return { hire, isHiring };
}

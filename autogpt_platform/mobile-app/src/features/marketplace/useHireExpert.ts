import { useRef, useState } from "react";

import {
  type Expert,
  type ExpertTemplate,
  expertKeys,
  hireExpert,
} from "@/lib/api/experts";
import { isSessionGone, queryClient, sessionGeneration, signOut } from "@/lib/session";

import { hireErrorMessage, withHiredExpert } from "./helpers";

interface Args {
  onHired: (expert: Expert, template: ExpertTemplate) => void;
  onError: (message: string, template: ExpertTemplate) => void;
}

export function useHireExpert({ onHired, onError }: Args) {
  const [hiringIDs, setHiringIDs] = useState<string[]>([]);
  // State lands a render late; the ref stops a second tap in the same frame.
  const inFlight = useRef(new Set<string>());

  async function hire(template: ExpertTemplate) {
    if (inFlight.current.has(template.id)) return;
    inFlight.current.add(template.id);
    setHiringIDs((ids) => [...ids, template.id]);
    const generation = sessionGeneration();
    try {
      const { expert } = await hireExpert(template.id);
      // Signed out meanwhile: the old account's expert stays out of the cache.
      if (generation !== sessionGeneration()) return;
      queryClient.setQueryData<Expert[]>(expertKeys.list, (old) =>
        withHiredExpert(old, expert),
      );
      void queryClient.invalidateQueries({ queryKey: expertKeys.list });
      onHired(expert, template);
    } catch (caught) {
      if (isSessionGone(caught)) {
        void signOut();
        return;
      }
      if (generation === sessionGeneration()) {
        onError(hireErrorMessage(caught, template.name), template);
      }
    } finally {
      inFlight.current.delete(template.id);
      setHiringIDs((ids) => ids.filter((id) => id !== template.id));
    }
  }

  function isHiring(templateID: string) {
    return hiringIDs.includes(templateID);
  }

  return { hire, isHiring };
}

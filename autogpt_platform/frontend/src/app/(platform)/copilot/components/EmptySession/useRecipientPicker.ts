import { parseAsString, useQueryState } from "nuqs";
import { useEffect } from "react";
import { useExpertMap } from "../../useExpertMap";
import type { RecipientOption } from "../ChatInput/components/RecipientChip";

const AUTOPILOT_RECIPIENT: RecipientOption = {
  id: null,
  name: "Autopilot",
  avatarUrl: null,
};

export function useRecipientPicker() {
  const {
    activeExperts,
    activeExpertIds,
    isExpertsEnabled,
    hasExpertsSettled,
  } = useExpertMap();
  const [expertIdParam, setExpertIdParam] = useQueryState(
    "expertId",
    parseAsString,
  );

  // An ?expertId= pointing at an expert the user can no longer address
  // (archived, deleted, or simply wrong) would leave the chip reading
  // "Autopilot" while `createSession` still sent the id — which the backend
  // rejects with a 404 on every send. Drop it so both agree on Autopilot.
  useEffect(
    function clearUnknownExpertParam() {
      if (!hasExpertsSettled || !expertIdParam) return;
      if (activeExpertIds.has(expertIdParam)) return;
      void setExpertIdParam(null);
    },
    [activeExpertIds, hasExpertsSettled, expertIdParam, setExpertIdParam],
  );

  const selectedExpert =
    activeExperts.find((expert) => expert.id === expertIdParam) ?? null;

  const options: RecipientOption[] = [
    AUTOPILOT_RECIPIENT,
    ...activeExperts.map((expert) => ({
      id: expert.id,
      name: expert.name,
      avatarUrl: expert.avatarUrl,
      color: expert.color,
    })),
  ];

  return {
    options,
    recipient:
      options.find((option) => option.id === expertIdParam) ??
      AUTOPILOT_RECIPIENT,
    selectedExpert,
    // Only a pending param can be mis-rendered as "Autopilot"; without one the
    // fallback is already the right answer. Keyed on "not settled yet" rather
    // than "fetching": an initial query that is pending but paused (offline)
    // reports `isFetching: false` while it still has no roster to resolve
    // against. Gated on the flag because with experts off the roster never
    // settles and the Autopilot fallback is the only correct answer.
    isLoadingRecipient:
      isExpertsEnabled && !hasExpertsSettled && !!expertIdParam,
    selectRecipient(id: string | null) {
      void setExpertIdParam(id);
    },
  };
}

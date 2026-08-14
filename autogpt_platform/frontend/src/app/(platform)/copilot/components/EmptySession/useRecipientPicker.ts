import { parseAsString, useQueryState } from "nuqs";
import { useEffect } from "react";
import { getActiveExperts, useExpertMap } from "../../useExpertMap";
import type { RecipientOption } from "../ChatInput/components/RecipientChip";

const AUTOPILOT_RECIPIENT: RecipientOption = {
  id: null,
  name: "Autopilot",
  avatarUrl: null,
};

export function useRecipientPicker() {
  const { expertsById, isLoadingExperts, hasLoadedExperts } = useExpertMap();
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
      if (!hasLoadedExperts || !expertIdParam) return;
      const expert = expertsById.get(expertIdParam);
      if (expert && !expert.isArchived) return;
      void setExpertIdParam(null);
    },
    [hasLoadedExperts, expertIdParam, expertsById, setExpertIdParam],
  );

  // The identity map includes archived experts (useExpertMap contract); only
  // the active roster is addressable from the picker.
  const options: RecipientOption[] = [
    AUTOPILOT_RECIPIENT,
    ...getActiveExperts(expertsById).map((expert) => ({
      id: expert.id,
      name: expert.name,
      avatarUrl: expert.avatarUrl,
    })),
  ];

  return {
    options,
    recipient:
      options.find((option) => option.id === expertIdParam) ??
      AUTOPILOT_RECIPIENT,
    // Only a pending param can be mis-rendered as "Autopilot"; without one the
    // fallback is already the right answer.
    isLoadingRecipient: isLoadingExperts && !!expertIdParam,
    selectRecipient(id: string | null) {
      void setExpertIdParam(id);
    },
  };
}

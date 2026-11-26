import { skipToken, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Alert } from "react-native";

import { useToast } from "@/components/useToast";
import { ApiError } from "@/lib/api/client";
import {
  type ExpertDetail,
  expertActivityKeys,
  getExpertDetail,
} from "@/lib/api/expertActivity";
import {
  type AutopilotMode,
  type ExpertSoul,
  updateExpertBudget,
  updateExpertMode,
  updateExpertSoul,
} from "@/lib/api/expertManage";
import { expertKeys } from "@/lib/api/experts";
import { flagKeys, getFlags, isFlagOn } from "@/lib/api/flags";

import { isExpertID, trimProfile, unsupervisedPrompt } from "./helpers";

const SAVE_FAILED = "Couldn't save. Try again.";

type Sheet = "profile" | "budget";

/** `routeExpertID` comes from the URL, which a link controls. */
export function useManageExpert(routeExpertID: unknown) {
  const expertID = isExpertID(routeExpertID) ? routeExpertID : null;
  const queryClient = useQueryClient();
  const { toast, show } = useToast();
  const [sheet, setSheet] = useState<Sheet | null>(null);
  const [isModeRefused, setIsModeRefused] = useState(false);
  const detailKey = expertActivityKeys.detail(expertID ?? "");
  const detail = useQuery({
    queryKey: detailKey,
    queryFn: expertID ? () => getExpertDetail(expertID) : skipToken,
  });
  const flags = useQuery({ queryKey: flagKeys.all, queryFn: getFlags });

  const isGone =
    expertID === null ||
    (detail.error instanceof ApiError && detail.error.status === 404);
  // A refetch that 404s keeps the old data; the expert is gone all the same.
  const expert = isGone ? null : (detail.data ?? null);

  // The page changes at once; a refused save puts it back as it was.
  async function save(
    patch: Partial<ExpertDetail>,
    request: (id: string) => Promise<ExpertDetail>,
  ) {
    if (!expertID) return false;
    await queryClient.cancelQueries({ queryKey: detailKey });
    const previous = queryClient.getQueryData<ExpertDetail>(detailKey);
    queryClient.setQueryData<ExpertDetail>(detailKey, (current) =>
      current ? { ...current, ...patch } : current,
    );
    try {
      queryClient.setQueryData(detailKey, await request(expertID));
      return true;
    } catch (caught) {
      queryClient.setQueryData(detailKey, previous);
      throw caught;
    } finally {
      void queryClient.invalidateQueries({ queryKey: expertKeys.list });
    }
  }

  // A sheet keeps the edits and shows the error itself: a toast would sit
  // behind it.
  async function saveProfile(form: ExpertSoul) {
    const soul = trimProfile(form);
    try {
      await save(soul, (id) => updateExpertSoul(id, soul));
      setSheet(null);
      return null;
    } catch {
      return SAVE_FAILED;
    }
  }

  async function saveBudget(credits: number | null) {
    try {
      await save({ weekly_budget: credits }, (id) => updateExpertBudget(id, credits));
      setSheet(null);
      return null;
    } catch {
      return SAVE_FAILED;
    }
  }

  async function saveMode(mode: AutopilotMode | null) {
    try {
      await save({ autopilot_mode: mode }, (id) => updateExpertMode(id, mode));
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 403) {
        setIsModeRefused(true);
        show({ message: "Approval modes aren't on for your account.", tone: "error" });
        return;
      }
      show({ message: SAVE_FAILED, tone: "error" });
    }
  }

  function chooseMode(mode: AutopilotMode | null) {
    if (!expert || mode === expert.autopilot_mode) return;
    if (mode !== "unsupervised") {
      void saveMode(mode);
      return;
    }
    const prompt = unsupervisedPrompt(expert.name);
    Alert.alert(prompt.title, prompt.body, [
      { text: "Cancel", style: "cancel" },
      {
        text: "Run unsupervised",
        style: "destructive",
        onPress: () => void saveMode("unsupervised"),
      },
    ]);
  }

  function openProfile() {
    setSheet("profile");
  }

  function openBudget() {
    setSheet("budget");
  }

  function closeSheet() {
    setSheet(null);
  }

  function retry() {
    void detail.refetch();
  }

  return {
    expert,
    isLoading: expertID !== null && detail.isPending,
    isGone,
    isLoadError: !expert && detail.isError && !isGone,
    retry,
    showApprovals: isFlagOn(flags.data, "copilot-auto-mode") && !isModeRefused,
    sheet,
    openProfile,
    openBudget,
    closeSheet,
    saveProfile,
    saveBudget,
    chooseMode,
    toast,
  };
}

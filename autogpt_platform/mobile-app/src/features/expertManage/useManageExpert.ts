import { skipToken, useQuery, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";
import { useState } from "react";
import { Alert } from "react-native";

import { useToast } from "@/components/useToast";
import { isSessionID } from "@/lib/api/chat";
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
import { type Expert, expertKeys } from "@/lib/api/experts";
import { flagKeys, getFlags, isFlagOn } from "@/lib/api/flags";

import { isExpertID, trimProfile, unsupervisedPrompt } from "./helpers";

const SAVE_FAILED = "Couldn't save. Try again.";

type Sheet = "profile" | "budget" | "reset" | "fire";

interface Args {
  /** From the URL, which a link controls. */
  routeExpertID: unknown;
  /** The chat whose panel opened this page; none means the Team page did. */
  routeFrom: unknown;
}

export function useManageExpert({ routeExpertID, routeFrom }: Args) {
  const expertID = isExpertID(routeExpertID) ? routeExpertID : null;
  const queryClient = useQueryClient();
  const { toast, show } = useToast();
  const [sheet, setSheet] = useState<Sheet | null>(null);
  const [isModeRefused, setIsModeRefused] = useState(false);
  // A reset removes the chat this page was opened from.
  const [isOriginGone, setIsOriginGone] = useState(false);
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

  // The page and the team list change at once; a refused save puts both
  // back as they were.
  async function save(
    patch: Partial<ExpertDetail>,
    request: (id: string) => Promise<ExpertDetail>,
  ) {
    if (!expertID) return false;
    const { name } = patch;
    await Promise.all([
      queryClient.cancelQueries({ queryKey: detailKey }),
      queryClient.cancelQueries({ queryKey: expertKeys.list, exact: true }),
    ]);
    const previous = queryClient.getQueryData<ExpertDetail>(detailKey);
    const previousList = queryClient.getQueryData<Expert[]>(expertKeys.list);
    queryClient.setQueryData<ExpertDetail>(detailKey, (current) =>
      current ? { ...current, ...patch } : current,
    );
    if (name !== undefined) {
      queryClient.setQueryData<Expert[]>(expertKeys.list, (list) =>
        list?.map((item) => (item.id === expertID ? { ...item, name } : item)),
      );
    }
    try {
      queryClient.setQueryData(detailKey, await request(expertID));
      return true;
    } catch (caught) {
      queryClient.setQueryData(detailKey, previous);
      if (name !== undefined) queryClient.setQueryData(expertKeys.list, previousList);
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

  function openReset() {
    setSheet("reset");
  }

  function openFire() {
    setSheet("fire");
  }

  function forgetOrigin() {
    setIsOriginGone(true);
  }

  function closeSheet() {
    setSheet(null);
  }

  function goBack() {
    if (isSessionID(routeFrom) && !isOriginGone) router.navigate(`/chat/${routeFrom}`);
    else router.navigate("/team");
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
    goBack,
    showApprovals: isFlagOn(flags.data, "copilot-auto-mode") && !isModeRefused,
    sheet,
    openProfile,
    openBudget,
    openReset,
    openFire,
    forgetOrigin,
    closeSheet,
    saveProfile,
    saveBudget,
    chooseMode,
    toast,
  };
}

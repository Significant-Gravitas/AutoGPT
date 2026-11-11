import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { PLAN_REQUIRED_MESSAGE } from "@/features/chat/chatErrors";
import { ApiError } from "@/lib/api/client";
import {
  expertActivityKeys,
  getExpertComputer,
  getExpertDetail,
  listExpertRuns,
  listFollowupSchedules,
  listGraphSchedules,
  resumeExpertSchedules,
} from "@/lib/api/expertActivity";
import { expertKeys } from "@/lib/api/experts";

import {
  computerState,
  expertSchedules,
  RECENT_LIMIT,
  UPCOMING_LIMIT,
} from "./helpers";

const RESUME_FAILED = "Couldn't resume routines. Try again.";

// Fetched only while the panel is open; each opening refreshes stale data.
export function useExpertActivity(expertID: string, isOpen: boolean) {
  const queryClient = useQueryClient();
  const [showAllUpcoming, setShowAllUpcoming] = useState(false);
  const detail = useQuery({
    queryKey: expertActivityKeys.detail(expertID),
    queryFn: () => getExpertDetail(expertID),
    enabled: isOpen,
  });
  const graphSchedules = useQuery({
    queryKey: expertActivityKeys.graphSchedules,
    queryFn: listGraphSchedules,
    enabled: isOpen,
  });
  const followupSchedules = useQuery({
    queryKey: expertActivityKeys.followupSchedules,
    queryFn: listFollowupSchedules,
    enabled: isOpen,
  });
  const runs = useQuery({
    queryKey: expertActivityKeys.runs(expertID),
    queryFn: () => listExpertRuns(expertID),
    enabled: isOpen,
  });
  const computer = useQuery({
    queryKey: expertActivityKeys.computer(expertID),
    queryFn: () => getExpertComputer(expertID),
    enabled: isOpen,
  });
  const resume = useMutation({
    mutationFn: () => resumeExpertSchedules(expertID),
    onSuccess(expert) {
      queryClient.setQueryData(expertActivityKeys.detail(expertID), expert);
      void queryClient.invalidateQueries({ queryKey: expertKeys.list });
      void queryClient.invalidateQueries({ queryKey: ["schedules"] });
    },
  });

  // A list that failed counts as empty, so one bad route doesn't hide the rest.
  const isUpcomingLoading =
    detail.isPending || graphSchedules.isPending || followupSchedules.isPending;
  const schedules = expertSchedules({
    expertID,
    detail: detail.data,
    graphSchedules: graphSchedules.data ?? [],
    followupSchedules: followupSchedules.data ?? [],
  });
  const isPaused = !!detail.data?.schedules_paused_at;
  const computerWord = computer.data ? computerState(computer.data) : null;
  const recent = (runs.data ?? []).slice(0, RECENT_LIMIT);

  function resumeErrorText() {
    if (!resume.isError) return null;
    if (resume.error instanceof ApiError && resume.error.status === 402) {
      return PLAN_REQUIRED_MESSAGE;
    }
    return RESUME_FAILED;
  }

  function showMoreUpcoming() {
    setShowAllUpcoming(true);
  }

  function resumeRoutines() {
    resume.mutate();
  }

  return {
    upcoming: {
      isLoading: isUpcomingLoading,
      isShown: !isUpcomingLoading && (schedules.length > 0 || isPaused),
      schedules: showAllUpcoming ? schedules : schedules.slice(0, UPCOMING_LIMIT),
      hiddenCount: showAllUpcoming ? 0 : Math.max(0, schedules.length - UPCOMING_LIMIT),
      showMore: showMoreUpcoming,
      isPaused,
      resume: resumeRoutines,
      isResuming: resume.isPending,
      resumeError: resumeErrorText(),
    },
    recent: {
      isLoading: runs.isPending,
      isShown: recent.length > 0,
      runs: recent,
    },
    computer: {
      isLoading: computer.isPending,
      isShown: computerWord !== null,
      state: computerWord,
    },
  };
}

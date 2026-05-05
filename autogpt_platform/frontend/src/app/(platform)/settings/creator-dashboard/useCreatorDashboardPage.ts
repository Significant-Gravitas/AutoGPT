import { useMemo, useState } from "react";
import * as Sentry from "@sentry/nextjs";

import {
  getGetV2ListMySubmissionsQueryKey,
  useDeleteV2DeleteStoreSubmission,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import type { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

import {
  applyFiltersAndSort,
  computeStats,
  INITIAL_FILTER_STATE,
  type FilterState,
} from "./helpers";

type PublishStep = "select" | "info" | "review";

interface PublishState {
  isOpen: boolean;
  step: PublishStep;
  submissionData: StoreSubmission | null;
}

interface EditPayload extends StoreSubmissionEditRequest {
  store_listing_version_id: string | undefined;
  graph_id: string;
}

interface EditState {
  isOpen: boolean;
  submission: EditPayload | null;
}

export function useCreatorDashboardPage() {
  const queryClient = getQueryClient();
  const { user } = useSupabase();

  const [publishState, setPublishState] = useState<PublishState>({
    isOpen: false,
    step: "select",
    submissionData: null,
  });

  const [editState, setEditState] = useState<EditState>({
    isOpen: false,
    submission: null,
  });

  const [filterState, setFilterState] =
    useState<FilterState>(INITIAL_FILTER_STATE);

  const {
    data: response,
    isSuccess,
    error,
    refetch,
  } = useGetV2ListMySubmissions(undefined, {
    query: {
      select: (x) => x.data as StoreSubmissionsResponse,
      enabled: !!user,
    },
  });

  const { mutateAsync: deleteSubmission } = useDeleteV2DeleteStoreSubmission({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListMySubmissionsQueryKey(),
        });
      },
    },
  });

  const submissions = response?.submissions ?? [];

  const stats = useMemo(() => computeStats(submissions), [submissions]);

  const visibleSubmissions = useMemo(
    () => applyFiltersAndSort(submissions, filterState),
    [submissions, filterState],
  );

  function resetFilters() {
    setFilterState(INITIAL_FILTER_STATE);
  }

  function openPublishModal() {
    setPublishState({
      isOpen: true,
      step: "select",
      submissionData: null,
    });
  }

  function onPublishStateChange(newState: PublishState) {
    setPublishState(newState);
  }

  function onViewSubmission(submission: StoreSubmission) {
    setPublishState({
      isOpen: true,
      step: "review",
      submissionData: submission,
    });
  }

  function onEditSubmission(submission: EditPayload) {
    setEditState({ isOpen: true, submission });
  }

  function onEditClose() {
    setEditState({ isOpen: false, submission: null });
  }

  async function onEditSuccess(submission: StoreSubmission) {
    try {
      if (!submission.listing_version_id) {
        Sentry.captureException(
          new Error("No store listing version ID found for submission"),
        );
        return;
      }
      setEditState({ isOpen: false, submission: null });
      await queryClient.invalidateQueries({
        queryKey: getGetV2ListMySubmissionsQueryKey(),
      });
    } catch (err) {
      Sentry.captureException(err);
    }
  }

  async function onDeleteSubmission(submissionId: string) {
    await deleteSubmission({ submissionId });
  }

  return {
    submissions,
    visibleSubmissions,
    stats,
    filterState,
    setFilterState,
    resetFilters,
    isLoading: !isSuccess,
    isReady: isSuccess,
    error,
    refetch,
    publishState,
    onPublishStateChange,
    openPublishModal,
    editState,
    onViewSubmission,
    onEditSubmission,
    onEditSuccess,
    onEditClose,
    onDeleteSubmission,
  };
}

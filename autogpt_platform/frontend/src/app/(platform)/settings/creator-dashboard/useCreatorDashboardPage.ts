import { useEffect, useMemo, useRef, useState } from "react";
import * as Sentry from "@sentry/nextjs";
import { keepPreviousData } from "@tanstack/react-query";

import {
  getGetV2ListMySubmissionsQueryKey,
  useDeleteV2DeleteStoreSubmission,
  useGetV2GetUserProfile,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import type { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";

import {
  applyFiltersAndSort,
  INITIAL_FILTER_STATE,
  toDashboardStats,
  type FilterState,
} from "./helpers";

const PAGE_SIZE = 20;

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

  const [filterState, setFilterStateRaw] =
    useState<FilterState>(INITIAL_FILTER_STATE);

  const [page, setPage] = useState(1);

  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: (x) => x.data as ProfileDetails,
      enabled: !!user,
    },
  });
  const creatorUsername = profile?.username;

  const {
    data: response,
    isSuccess,
    isFetching,
    error,
    refetch,
  } = useGetV2ListMySubmissions(
    { page, page_size: PAGE_SIZE },
    {
      query: {
        select: (x) => x.data as StoreSubmissionsResponse,
        enabled: !!user,
        placeholderData: keepPreviousData,
      },
    },
  );

  const [isSortingTransition, setIsSortingTransition] = useState(false);
  const sortingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (sortingTimerRef.current) clearTimeout(sortingTimerRef.current);
    };
  }, []);

  function setFilterState(next: FilterState) {
    setFilterStateRaw(next);
    setPage(1);
    setIsSortingTransition(true);
    if (sortingTimerRef.current) clearTimeout(sortingTimerRef.current);
    sortingTimerRef.current = setTimeout(() => {
      setIsSortingTransition(false);
    }, 400);
  }

  function onPageChange(nextPage: number) {
    setPage(nextPage);
  }

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

  const stats = toDashboardStats(response?.stats);

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
    pagination: response?.pagination,
    onPageChange,
    isFetching: isFetching || isSortingTransition,
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
    creatorUsername,
  };
}

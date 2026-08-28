import { useEffect, useRef, useState } from "react";
import * as Sentry from "@sentry/nextjs";
import { keepPreviousData } from "@tanstack/react-query";

import {
  getGetV2ListMySubmissionsQueryKey,
  deleteV2DeleteStoreSubmission,
  useGetV2GetUserProfile,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
import { useDebouncedValue } from "@/hooks/useDebouncedValue";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import type { PublishState } from "@/components/contextual/PublishAgentModal/usePublishAgentModal";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";

import {
  buildEditPayload,
  INITIAL_FILTER_STATE,
  toDashboardStats,
  type EditPayload,
  type FilterState,
} from "./helpers";

const PAGE_SIZE = 20;
const SEARCH_QUERY_MAX_LENGTH = 100;

interface EditState {
  isOpen: boolean;
  submission: EditPayload | null;
}

export function useCreatorDashboardPage() {
  const queryClient = getQueryClient();
  const { user } = useAuth();
  const activeOrgID = useOrgTeamStore((state) => state.activeOrgID);
  const activeTeamID = useOrgTeamStore((state) => state.activeTeamID);
  const isTenantReady = useOrgTeamStore((state) => state.isLoaded);

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
  const [searchInput, setSearchInput] = useState("");
  const [searchResetPending, setSearchResetPending] = useState(false);
  const debouncedSearch = useDebouncedValue(searchInput.trim(), 300);
  const isDebouncingSearch = searchInput.trim() !== debouncedSearch;
  const queryPage = searchResetPending ? 1 : page;

  function handleSearchChange(next: string) {
    const cappedNext = next.slice(0, SEARCH_QUERY_MAX_LENGTH);
    const prevTrimmed = searchInput.trim();
    setSearchInput(cappedNext);
    if (cappedNext.trim() !== prevTrimmed) {
      setSearchResetPending(true);
    }
  }

  useEffect(() => {
    if (!searchResetPending || isDebouncingSearch) return;
    setPage(1);
    setSearchResetPending(false);
  }, [isDebouncingSearch, searchResetPending]);

  const { data: profile } = useGetV2GetUserProfile({
    query: {
      select: (x) => x.data as ProfileDetails,
      enabled: !!user,
    },
  });
  const creatorUsername = profile?.username;

  const submissionParams = {
    page: queryPage,
    page_size: PAGE_SIZE,
    search_query: debouncedSearch || undefined,
    statuses:
      filterState.statuses.length > 0
        ? filterState.statuses.join(",")
        : undefined,
    sort_key: filterState.sortKey ?? undefined,
    sort_dir: filterState.sortKey ? filterState.sortDir : undefined,
  };

  const {
    data: response,
    isSuccess,
    isFetching,
    error,
    refetch,
  } = useGetV2ListMySubmissions(submissionParams, {
    query: {
      select: (x) => x.data as StoreSubmissionsResponse,
      enabled: !!user && isTenantReady && !isDebouncingSearch,
      queryKey: getTeamScopedQueryKey(
        getGetV2ListMySubmissionsQueryKey(submissionParams),
        activeOrgID,
        activeTeamID,
      ),
      placeholderData: keepPreviousData,
    },
    request: getTenantRequestInit(activeOrgID, activeTeamID, isTenantReady),
  });

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

  const submissions = response?.submissions ?? [];

  const stats = toDashboardStats(response?.stats);

  const visibleSubmissions = submissions;

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

  function onEditFromReview(submission: StoreSubmission) {
    if (!submission.listing_version_id) {
      Sentry.captureException(
        new Error("No store listing version ID found for submission"),
      );
      return;
    }
    setPublishState({ isOpen: false, step: "select", submissionData: null });
    onEditSubmission(buildEditPayload(submission));
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

  async function onDeleteSubmission(submission: StoreSubmission) {
    await deleteV2DeleteStoreSubmission(
      submission.listing_version_id,
      getTenantRequestInit(submission.organization_id, submission.team_id),
    );
    await queryClient.invalidateQueries({
      queryKey: getGetV2ListMySubmissionsQueryKey(),
    });
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
    searchInput,
    setSearchInput: handleSearchChange,
    debouncedSearch,
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
    onEditFromReview,
    onEditSuccess,
    onEditClose,
    onDeleteSubmission,
    creatorUsername,
  };
}

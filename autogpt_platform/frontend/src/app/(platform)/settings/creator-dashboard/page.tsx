"use client";

import { useEffect } from "react";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { EditAgentModal } from "@/components/contextual/EditAgentModal/EditAgentModal";

import { DashboardHeader } from "./components/DashboardHeader/DashboardHeader";
import { DashboardSkeleton } from "./components/DashboardSkeleton/DashboardSkeleton";
import { EmptyState } from "./components/EmptyState/EmptyState";
import { MobileSubmissionsList } from "./components/MobileSubmissionsList/MobileSubmissionsList";
import { StatsOverview } from "./components/StatsOverview/StatsOverview";
import { SubmissionsList } from "./components/SubmissionsList/SubmissionsList";
import { useCreatorDashboardPage } from "./useCreatorDashboardPage";

export default function SettingsCreatorDashboardPage() {
  useEffect(() => {
    document.title = "Creator dashboard – AutoGPT Platform";
  }, []);

  const {
    submissions,
    visibleSubmissions,
    pagination,
    onPageChange,
    isFetching,
    stats,
    filterState,
    setFilterState,
    resetFilters,
    isLoading,
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
  } = useCreatorDashboardPage();

  if (error) {
    return (
      <ErrorCard
        context="creator dashboard"
        responseError={
          error ? { detail: (error as { detail?: string }).detail } : undefined
        }
        onRetry={() => {
          void refetch();
        }}
      />
    );
  }

  if (isLoading) {
    return <DashboardSkeleton />;
  }

  const isEmpty = submissions.length === 0;

  return (
    <div className="flex flex-col gap-6 pb-8">
      <DashboardHeader
        publishState={publishState}
        onPublishStateChange={onPublishStateChange}
        onOpenSubmit={openPublishModal}
      />

      {isEmpty ? (
        <EmptyState />
      ) : (
        <>
          <StatsOverview stats={stats} index={0} />
          <div className="hidden md:block">
            <SubmissionsList
              submissions={visibleSubmissions}
              totalCount={pagination?.total_items ?? submissions.length}
              pagination={pagination}
              onPageChange={onPageChange}
              isFetching={isFetching}
              filterState={filterState}
              onFilterChange={setFilterState}
              onResetFilters={resetFilters}
              onView={onViewSubmission}
              onEdit={onEditSubmission}
              onDelete={onDeleteSubmission}
              creatorUsername={creatorUsername}
              index={2}
            />
          </div>
          <div className="md:hidden">
            <MobileSubmissionsList
              submissions={visibleSubmissions}
              totalCount={pagination?.total_items ?? submissions.length}
              pagination={pagination}
              onPageChange={onPageChange}
              isFetching={isFetching}
              filterState={filterState}
              onFilterChange={setFilterState}
              onResetFilters={resetFilters}
              onView={onViewSubmission}
              onEdit={onEditSubmission}
              onDelete={onDeleteSubmission}
              creatorUsername={creatorUsername}
            />
          </div>
        </>
      )}

      <EditAgentModal
        isOpen={editState.isOpen}
        onClose={onEditClose}
        submission={editState.submission}
        onSuccess={onEditSuccess}
      />
    </div>
  );
}

import {
  getGetV2ListMySubmissionsQueryKey,
  deleteV2DeleteStoreSubmission,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useState } from "react";
import * as Sentry from "@sentry/nextjs";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";

type PublishStep = "select" | "info" | "review";

type PublishState = {
  isOpen: boolean;
  step: PublishStep;
  submissionData: StoreSubmission | null;
};

type EditState = {
  isOpen: boolean;
  submission:
    | (StoreSubmissionEditRequest & {
        store_listing_version_id: string | undefined;
        graph_id: string;
        organization_id: string | null;
        team_id: string | null;
      })
    | null;
};

export const useMainDashboardPage = () => {
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

  const {
    data: submissions,
    isSuccess,
    error,
  } = useGetV2ListMySubmissions(undefined, {
    query: {
      select: (x) => {
        return x.data as StoreSubmissionsResponse;
      },
      enabled: !!user && isTenantReady,
      queryKey: getTeamScopedQueryKey(
        getGetV2ListMySubmissionsQueryKey(),
        activeOrgID,
        activeTeamID,
      ),
    },
    request: getTenantRequestInit(activeOrgID, activeTeamID, isTenantReady),
  });

  const onViewSubmission = (submission: StoreSubmission) => {
    setPublishState({
      isOpen: true,
      step: "review",
      submissionData: submission,
    });
  };

  const onEditSubmission = (
    submission: StoreSubmissionEditRequest & {
      store_listing_version_id: string | undefined;
      graph_id: string;
      organization_id: string | null;
      team_id: string | null;
    },
  ) => {
    setEditState({
      isOpen: true,
      submission,
    });
  };

  const onEditSuccess = async (submission: StoreSubmission) => {
    try {
      if (!submission.listing_version_id) {
        Sentry.captureException(
          new Error("No store listing version ID found for submission"),
        );
        return;
      }

      setEditState({
        isOpen: false,
        submission: null,
      });
    } catch (error) {
      Sentry.captureException(error);
    }
  };

  const onEditClose = () => {
    setEditState({
      isOpen: false,
      submission: null,
    });
  };

  const onDeleteSubmission = async (submission: StoreSubmission) => {
    await deleteV2DeleteStoreSubmission(
      submission.listing_version_id,
      getTenantRequestInit(submission.organization_id, submission.team_id),
    );
    await queryClient.invalidateQueries({
      queryKey: getGetV2ListMySubmissionsQueryKey(),
    });
  };

  const onOpenSubmitModal = () => {
    // Always reset to clean state when opening for new submission
    setPublishState({
      isOpen: true,
      step: "select",
      submissionData: null,
    });
  };

  const onPublishStateChange = (newState: PublishState) => {
    setPublishState(newState);
  };

  return {
    onOpenSubmitModal,
    onPublishStateChange,
    onDeleteSubmission,
    onViewSubmission,
    onEditSubmission,
    onEditSuccess,
    onEditClose,
    publishState,
    editState,
    // API data
    submissions,
    isLoading: !isSuccess,
    error,
  };
};

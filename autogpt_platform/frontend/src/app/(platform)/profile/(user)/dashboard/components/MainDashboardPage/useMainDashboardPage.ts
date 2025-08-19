import {
  getGetV2ListMySubmissionsQueryKey,
  useDeleteV2DeleteStoreSubmission,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useState } from "react";
import * as Sentry from "@sentry/nextjs";

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
        agent_id: string;
      })
    | null;
};

export const useMainDashboardPage = () => {
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

  const { mutateAsync: deleteSubmission } = useDeleteV2DeleteStoreSubmission({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListMySubmissionsQueryKey(),
        });
      },
    },
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
      enabled: !!user,
    },
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
      agent_id: string;
    },
  ) => {
    setEditState({
      isOpen: true,
      submission,
    });
  };

  const onEditSuccess = async (submission: StoreSubmission) => {
    try {
      if (!submission.store_listing_version_id) {
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

  const onDeleteSubmission = async (submission_id: string) => {
    await deleteSubmission({
      submissionId: submission_id,
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

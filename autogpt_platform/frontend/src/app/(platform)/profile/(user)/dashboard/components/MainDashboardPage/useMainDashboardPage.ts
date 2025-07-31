import {
  getGetV2ListMySubmissionsQueryKey,
  useDeleteV2DeleteStoreSubmission,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useState } from "react";

type PublishStep = "select" | "info" | "review";

type PublishState = {
  isOpen: boolean;
  step: PublishStep;
  submissionData: StoreSubmission | null;
};

export const useMainDashboardPage = () => {
  const queryClient = getQueryClient();

  const { user } = useSupabase();

  const [publishState, setPublishState] = useState<PublishState>({
    isOpen: false,
    step: "select",
    submissionData: null,
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
    publishState,
    // API data
    submissions,
    isLoading: !isSuccess,
    error,
  };
};

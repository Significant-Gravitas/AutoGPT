import {
  useDeleteV2DeleteStoreSubmission,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import { StoreSubmissionRequest } from "@/app/api/__generated__/models/storeSubmissionRequest";
import { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useState } from "react";

export const useMainDashboardPage = () => {
  const { user } = useSupabase();
  const [openPopout, setOpenPopout] = useState<boolean>(false);
  const [submissionData, setSubmissionData] =
    useState<StoreSubmissionRequest>();
  const [popoutStep, setPopoutStep] = useState<"select" | "info" | "review">(
    "info",
  );

  const { mutateAsync: deleteSubmission } = useDeleteV2DeleteStoreSubmission();

  const { data: submissions, isLoading } = useGetV2ListMySubmissions(
    undefined,
    {
      query: {
        select: (x) => {
          return x.data as StoreSubmissionsResponse;
        },
        enabled: !!user,
      },
    },
  );

  const onEditSubmission = (submission: StoreSubmissionRequest) => {
    setSubmissionData(submission);
    setPopoutStep("review");
    setOpenPopout(true);
  };

  const onDeleteSubmission = async (submission_id: string) => {
    await deleteSubmission({
      submissionId: submission_id,
    });
  };

  const onOpenPopout = () => {
    setPopoutStep("select");
    setOpenPopout(true);
  };

  return {
    onOpenPopout,
    onDeleteSubmission,
    onEditSubmission,
    submissions,
    isLoading,
    openPopout,
    submissionData,
    popoutStep,
  };
};

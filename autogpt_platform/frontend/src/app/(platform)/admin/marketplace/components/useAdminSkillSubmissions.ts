import {
  getGetV2AdminListPendingSkillSubmissionsQueryKey,
  useGetV2AdminListPendingSkillSubmissions,
  usePostV2ReviewSkillSubmission,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

export function useAdminSkillSubmissions() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const query = useGetV2AdminListPendingSkillSubmissions({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });

  const { mutate: review, isPending } = usePostV2ReviewSkillSubmission({
    mutation: {
      onSuccess: () =>
        queryClient.invalidateQueries({
          queryKey: getGetV2AdminListPendingSkillSubmissionsQueryKey(),
        }),
      onError: (error) =>
        toast({
          title: "Couldn't record that review",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        }),
    },
  });

  function submitReview(versionId: string, isApproved: boolean) {
    review({
      skillListingVersionId: versionId,
      data: {
        store_listing_version_id: versionId,
        is_approved: isApproved,
        comments: isApproved ? "Approved" : "Rejected",
      },
    });
  }

  return {
    submissions: query.data ?? [],
    isLoading: query.isLoading,
    isReviewing: isPending,
    approve: (versionId: string) => submitReview(versionId, true),
    reject: (versionId: string) => submitReview(versionId, false),
  };
}

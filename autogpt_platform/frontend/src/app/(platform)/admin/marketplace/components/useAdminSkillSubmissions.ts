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

  const query = useGetV2AdminListPendingSkillSubmissions();

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
        is_approved: isApproved,
        comments: isApproved ? "Approved" : "Rejected",
      },
    });
  }

  // A 404 here is the skills-hub flag being off, not an empty queue — reporting
  // "nothing to review" would tell an admin the opposite of the truth.
  const loaded = query.data?.status === 200 ? query.data.data : null;

  return {
    submissions: loaded ?? [],
    isLoading: query.isLoading,
    isUnavailable: !query.isLoading && loaded === null,
    isReviewing: isPending,
    approve: (versionId: string) => submitReview(versionId, true),
    reject: (versionId: string) => submitReview(versionId, false),
  };
}

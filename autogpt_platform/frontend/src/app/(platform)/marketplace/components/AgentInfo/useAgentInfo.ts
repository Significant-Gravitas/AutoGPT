import { usePostV2AddMarketplaceAgent } from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import * as Sentry from "@sentry/nextjs";
import { useGetV2DownloadAgentFile } from "@/app/api/__generated__/endpoints/store/store";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";

interface UseAgentInfoProps {
  storeListingVersionId: string;
}

export const useAgentInfo = ({ storeListingVersionId }: UseAgentInfoProps) => {
  const { toast } = useToast();
  const router = useRouter();
  const { completeStep } = useOnboarding();

  const {
    mutate: addMarketplaceAgentToLibrary,
    isPending: isAddingAgentToLibrary,
  } = usePostV2AddMarketplaceAgent({
    mutation: {
      onSuccess: ({ data }) => {
        completeStep("MARKETPLACE_ADD_AGENT");
        router.push(`/library/agents/${(data as LibraryAgent).id}`);
        toast({
          title: "Agent Added",
          description: "Redirecting to your library...",
          duration: 2000,
        });
      },
      onError: (error) => {
        Sentry.captureException(error);
        toast({
          title: "Error",
          description: "Failed to add agent to library. Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  const { refetch: downloadAgent, isFetching: isDownloadingAgent } =
    useGetV2DownloadAgentFile(storeListingVersionId, {
      query: {
        enabled: false,
        select: (data) => {
          return data.data;
        },
      },
    });

  const handleLibraryAction = async () => {
    addMarketplaceAgentToLibrary({
      data: { store_listing_version_id: storeListingVersionId },
    });
  };

  const handleDownload = async () => {
    try {
      const { data: file } = await downloadAgent();

      const jsonData = JSON.stringify(file, null, 2);
      const blob = new Blob([jsonData], { type: "application/json" });
      const url = window.URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = `agent_${storeListingVersionId}.json`;

      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      window.URL.revokeObjectURL(url);

      toast({
        title: "Download Complete",
        description: "Your agent has been successfully downloaded.",
      });
    } catch (error) {
      Sentry.captureException(error);
      toast({
        title: "Error",
        description: "Failed to download agent. Please try again.",
        variant: "destructive",
      });
    }
  };

  return {
    isAddingAgentToLibrary,
    handleLibraryAction,
    handleDownload,
    isDownloadingAgent,
  };
};

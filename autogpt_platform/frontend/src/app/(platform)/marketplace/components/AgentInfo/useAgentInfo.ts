import {
  getGetV2ListLibraryAgentsQueryKey,
  usePostV2AddMarketplaceAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import * as Sentry from "@sentry/nextjs";
import { useGetV2DownloadAgentFile } from "@/app/api/__generated__/endpoints/store/store";
import { analytics } from "@/services/analytics";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { agentGraphExportFilename, exportAsJSONFile } from "@/lib/utils";
import { useQueryClient } from "@tanstack/react-query";

interface UseAgentInfoProps {
  storeListingVersionId: string;
}

export const useAgentInfo = ({ storeListingVersionId }: UseAgentInfoProps) => {
  const { toast } = useToast();
  const router = useRouter();
  const queryClient = useQueryClient();

  const {
    mutateAsync: addMarketplaceAgentToLibrary,
    isPending: isAddingAgentToLibrary,
  } = usePostV2AddMarketplaceAgent();

  const { refetch: downloadAgent, isFetching: isDownloadingAgent } =
    useGetV2DownloadAgentFile(storeListingVersionId, {
      query: {
        enabled: false,
        select: (data) => {
          return data.data;
        },
      },
    });

  const handleLibraryAction = async ({
    isAddingAgentFirstTime,
  }: {
    isAddingAgentFirstTime: boolean;
  }) => {
    try {
      const { data: response } = await addMarketplaceAgentToLibrary({
        data: { store_listing_version_id: storeListingVersionId },
      });

      const data = response as LibraryAgent;

      if (isAddingAgentFirstTime) {
        await queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryAgentsQueryKey(),
        });

        analytics.sendDatafastEvent("add_to_library", {
          name: data.name,
          id: data.id,
        });
      }

      router.push(`/library/agents/${data.id}`);

      toast({
        title: "Agent Added",
        description: "Redirecting to your library...",
        duration: 2000,
      });
    } catch (error) {
      Sentry.captureException(error);

      toast({
        title: "Error",
        description: "Failed to add agent to library. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleDownload = async (agentId: string, agentName: string) => {
    try {
      const { data: file } = await downloadAgent();

      exportAsJSONFile(
        file as object,
        agentGraphExportFilename(file, agentName),
      );

      analytics.sendDatafastEvent("download_agent", {
        name: agentName,
        id: agentId,
      });

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

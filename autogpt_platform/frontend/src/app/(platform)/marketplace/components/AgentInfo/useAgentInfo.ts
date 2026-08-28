import { useToast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import * as Sentry from "@sentry/nextjs";
import { useGetV2DownloadAgentFile } from "@/app/api/__generated__/endpoints/store/store";
import { analytics } from "@/services/analytics";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { agentGraphExportFilename, exportAsJSONFile } from "@/lib/utils";
import { useOrgTeamStore } from "@/services/org-team/store";
import { getLibraryAgentHref } from "@/services/org-team/builder";

interface UseAgentInfoProps {
  storeListingVersionId: string;
  libraryAgent?: LibraryAgent;
}

export const useAgentInfo = ({
  storeListingVersionId,
  libraryAgent,
}: UseAgentInfoProps) => {
  const { toast } = useToast();
  const router = useRouter();
  const activeOrgID = useOrgTeamStore((state) => state.activeOrgID);
  const activeTeamID = useOrgTeamStore((state) => state.activeTeamID);

  const { refetch: downloadAgent, isFetching: isDownloadingAgent } =
    useGetV2DownloadAgentFile(storeListingVersionId, {
      query: {
        enabled: false,
        select: (data) => {
          return data.data;
        },
      },
    });

  const handleOpenLibraryAgent = () => {
    if (!libraryAgent) return;
    router.push(
      getLibraryAgentHref(
        libraryAgent.id,
        libraryAgent.organization_id ?? activeOrgID ?? null,
        libraryAgent.team_id ?? activeTeamID ?? null,
      ),
    );
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
    handleOpenLibraryAgent,
    handleDownload,
    isDownloadingAgent,
  };
};

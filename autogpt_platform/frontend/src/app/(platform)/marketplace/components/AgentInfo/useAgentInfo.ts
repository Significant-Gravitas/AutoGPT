import { usePostV2AddMarketplaceAgent } from "@/app/api/__generated__/endpoints/library/library";
import { useGetV2DownloadAgentFile } from "@/app/api/__generated__/endpoints/store/store";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import { useToast } from "@/components/ui/use-toast";
import { useRouter } from "next/navigation";

interface useAgentInfoProps {
  storeListingVersionId: string;
  libraryAgent: LibraryAgent | null | undefined;
}

export const useAgentInfo = ({
  storeListingVersionId,
  libraryAgent,
}: useAgentInfoProps) => {
  const router = useRouter();
  const { toast } = useToast();

  const { completeStep } = useOnboarding();

  const { mutateAsync: addAgentToLibrary, isPending: adding } =
    usePostV2AddMarketplaceAgent({
      mutation: {
        onSuccess: ({ data }) => {
          const newLibraryAgent = data as LibraryAgent;
          console.log(newLibraryAgent);
          completeStep("MARKETPLACE_ADD_AGENT");
          router.push(`/library/agents/${newLibraryAgent.id}`);
          toast({
            title: "Agent Added",
            description: "Redirecting to your library...",
            duration: 2000,
          });
        },
        onError: (error) => {
          console.error("Failed to add agent to library:", error);
          toast({
            title: "EaddAgentToLibraryrror",
            description: "Failed to add agent to library. Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  const { isLoading: downloading, refetch: downloadAgentFile } =
    useGetV2DownloadAgentFile(storeListingVersionId, {
      query: {
        enabled: false,
        select: (x) => {
          return x.data;
        },
      },
    });

  const libraryAction = async () => {
    if (libraryAgent) {
      toast({
        description: "Redirecting to your library...",
        duration: 2000,
      });
      router.push(`/library/agents/${libraryAgent.id}`);
      return;
    }

    await addAgentToLibrary({
      data: {
        store_listing_version_id: storeListingVersionId,
      },
    });
  };

  const handleDownload = async () => {
    const downloadAgent = async (): Promise<void> => {
      try {
        const { data: file } = await downloadAgentFile();
        // Similar to Marketplace v1
        const jsonData = JSON.stringify(file, null, 2);
        // Create a Blob from the file content
        const blob = new Blob([jsonData], { type: "application/json" });

        // Create a temporary URL for the Blob
        const url = window.URL.createObjectURL(blob);

        // Create a temporary anchor element
        const a = document.createElement("a");
        a.href = url;
        a.download = `agent_${storeListingVersionId}.json`; // Set the filename

        // Append the anchor to the body, click it, and remove it
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // Revoke the temporary URL
        window.URL.revokeObjectURL(url);

        toast({
          title: "Download Complete",
          description: "Your agent has been successfully downloaded.",
        });
      } catch (error) {
        console.error(`Error downloading agent:`, error);
        toast({
          title: "Error",
          description: "Failed to download agent. Please try again.",
          variant: "destructive",
        });
      }
    };
    await downloadAgent();
  };

  return {
    adding,
    downloading,
    libraryAction,
    handleDownload,
  };
};

import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import { useToast } from "@/components/ui/use-toast";
import BackendAPI, { LibraryAgent } from "@/lib/autogpt-server-api";
import { useRouter } from "next/navigation";
import { useState } from "react";

interface useAgentInfoProps {
  storeListingVersionId: string;
  libraryAgent: LibraryAgent | null;
}

export const useAgentInfo = ({
  storeListingVersionId,
  libraryAgent,
}: useAgentInfoProps) => {
  const router = useRouter();
  const api = new BackendAPI();
  const { toast } = useToast();

  const { completeStep } = useOnboarding();

  const [adding, setAdding] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const libraryAction = async () => {
    setAdding(true);
    if (libraryAgent) {
      toast({
        description: "Redirecting to your library...",
        duration: 2000,
      });
      // Redirect to the library agent page
      router.push(`/library/agents/${libraryAgent.id}`);
      return;
    }
    try {
      const newLibraryAgent = await api.addMarketplaceAgentToLibrary(
        storeListingVersionId,
      );
      completeStep("MARKETPLACE_ADD_AGENT");
      router.push(`/library/agents/${newLibraryAgent.id}`);
      toast({
        title: "Agent Added",
        description: "Redirecting to your library...",
        duration: 2000,
      });
    } catch (error) {
      console.error("Failed to add agent to library:", error);
      toast({
        title: "Error",
        description: "Failed to add agent to library. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleDownload = async () => {
    const downloadAgent = async (): Promise<void> => {
      setDownloading(true);
      try {
        const file = await api.downloadStoreAgent(storeListingVersionId);

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
    setDownloading(false);
  };

  return {
    adding,
    downloading,
    libraryAction,
    handleDownload,
  };
};

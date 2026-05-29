import { useState, useEffect } from "react";
import {
  usePostV1EnableExecutionSharing,
  useDeleteV1DisableExecutionSharing,
} from "@/app/api/__generated__/endpoints/default/default";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { executionShareUrl } from "@/lib/share/routes";

interface UseShareRunButtonProps {
  graphId: string;
  executionId: string;
  isShared?: boolean;
  shareToken?: string | null;
}

export function useShareRunButton({
  graphId,
  executionId,
  isShared: initialIsShared = false,
  shareToken: initialShareToken,
}: UseShareRunButtonProps) {
  const [isShared, setIsShared] = useState(initialIsShared);
  const [shareToken, setShareToken] = useState(initialShareToken || null);
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  // Sync state when props change (e.g., after re-fetching run data)
  useEffect(() => {
    setIsShared(initialIsShared);
    setShareToken(initialShareToken || null);
  }, [initialIsShared, initialShareToken]);

  const shareUrl = shareToken ? executionShareUrl(shareToken) : "";

  const { mutateAsync: enableSharing, isPending: isEnabling } =
    usePostV1EnableExecutionSharing();
  const { mutateAsync: disableSharing, isPending: isDisabling } =
    useDeleteV1DisableExecutionSharing();

  const loading = isEnabling || isDisabling;

  async function handleShare() {
    try {
      const response = await enableSharing({
        graphId,
        graphExecId: executionId,
        data: {}, // Empty ShareRequest
      });

      if (response.status === 200) {
        setShareToken(response.data.share_token);
        setIsShared(true);
        toast({
          title: "Sharing enabled",
          description:
            "Your agent run is now publicly accessible via the share link.",
        });
      } else {
        toast({
          title: "Error",
          description: "Failed to enable sharing. Please try again.",
          variant: "destructive",
        });
      }
    } catch {
      toast({
        title: "Error",
        description: "Failed to enable sharing. Please try again.",
        variant: "destructive",
      });
    }
  }

  async function handleStopSharing() {
    try {
      await disableSharing({
        graphId,
        graphExecId: executionId,
      });

      setIsShared(false);
      setShareToken(null);
      toast({
        title: "Sharing disabled",
        description: "The share link is no longer accessible.",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to disable sharing. Please try again.",
        variant: "destructive",
      });
    }
  }

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast({
        title: "Copied!",
        description: "Share link copied to clipboard.",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to copy link. Please try again.",
        variant: "destructive",
      });
    }
  }

  return {
    isShared,
    shareToken,
    shareUrl,
    copied,
    loading,
    handleShare,
    handleStopSharing,
    handleCopy,
  };
}

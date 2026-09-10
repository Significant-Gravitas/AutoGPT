import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  getGetV2GetChatShareStateQueryKey,
  useDeleteV2DisableChatSharing,
  useGetV2GetChatShareState,
  usePostV2EnableChatSharing,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { chatShareUrl } from "@/lib/share/routes";

type Props = {
  sessionId: string;
  open: boolean;
};

export function useShareChatDialog({ sessionId, open }: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [isShared, setIsShared] = useState(false);
  const [shareToken, setShareToken] = useState<string | null>(null);
  // Default ON so the common case ("share this chat and the runs it
  // produced") works without the owner having to think about it.
  // Server state overrides this once the share-state query resolves.
  const [autoShareExecutions, setAutoShareExecutions] = useState(true);
  const [copied, setCopied] = useState(false);
  // Two-step confirmation for "Stop sharing".  A share may have been
  // posted to social / email / Slack — an accidental click would dead-
  // letter every external reference with no undo (the next re-enable
  // mints a new token).  Flipped on first click; second click is the
  // authoritative DELETE.
  const [confirmingStop, setConfirmingStop] = useState(false);

  const { data: stateResponse, isLoading: isLoadingState } =
    useGetV2GetChatShareState(sessionId, {
      query: {
        enabled: open,
        select: (res) => (res.status === 200 ? res.data : undefined),
      },
    });

  // Hydrate local state from the backend payload so the modal opens
  // in the right mode after a reload.
  useEffect(() => {
    if (stateResponse) {
      setIsShared(stateResponse.is_shared ?? false);
      setShareToken(stateResponse.share_token ?? null);
      if (stateResponse.is_shared) {
        setAutoShareExecutions(stateResponse.auto_share_executions ?? false);
      }
    }
  }, [stateResponse]);

  const invalidateState = () =>
    queryClient.invalidateQueries({
      queryKey: getGetV2GetChatShareStateQueryKey(sessionId),
    });

  const { mutate: enable, isPending: isEnabling } = usePostV2EnableChatSharing({
    mutation: {
      onSuccess: (res) => {
        if (res.status !== 200) {
          toast({
            title: "Failed to enable sharing",
            description: "Please try again.",
            variant: "destructive",
          });
          return;
        }
        setIsShared(true);
        setShareToken(res.data.share_token);
        invalidateState();
        toast({
          title: "Chat sharing enabled",
          description:
            "Anyone with the link can now view this conversation. Revoke any time.",
        });
      },
      onError: () => {
        toast({
          title: "Failed to enable sharing",
          description: "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  const { mutate: disable, isPending: isDisabling } =
    useDeleteV2DisableChatSharing({
      mutation: {
        onSuccess: () => {
          setIsShared(false);
          setShareToken(null);
          setConfirmingStop(false);
          invalidateState();
          toast({
            title: "Chat sharing disabled",
            description: "The share link is no longer accessible.",
          });
        },
        onError: () => {
          toast({
            title: "Failed to disable sharing",
            description: "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  const shareUrl = shareToken ? chatShareUrl(shareToken) : "";

  async function copyShareUrl() {
    if (!shareUrl) return;
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast({
        title: "Failed to copy link",
        variant: "destructive",
      });
    }
  }

  // Disclosure counts surfaced to the modal so the owner sees what
  // they're about to expose.  Defaults to 0 while the share-state
  // query is loading.  Values from the latest server snapshot.
  const messageCount = stateResponse?.message_count ?? 0;
  const linkedRunCount = stateResponse?.linked_run_count ?? 0;
  const fileCount = stateResponse?.file_count ?? 0;

  function requestStop() {
    if (!confirmingStop) {
      setConfirmingStop(true);
      return;
    }
    disable({ sessionId });
  }

  function cancelStop() {
    setConfirmingStop(false);
  }

  return {
    isShared,
    shareToken,
    shareUrl,
    copied,
    autoShareExecutions,
    setAutoShareExecutions,
    isLoadingState,
    enable: () =>
      enable({
        sessionId,
        data: { auto_share_executions: autoShareExecutions },
      }),
    isEnabling,
    requestStop,
    cancelStop,
    confirmingStop,
    isDisabling,
    copyShareUrl,
    messageCount,
    linkedRunCount,
    fileCount,
  };
}

export type ShareChatDialogState = ReturnType<typeof useShareChatDialog>;

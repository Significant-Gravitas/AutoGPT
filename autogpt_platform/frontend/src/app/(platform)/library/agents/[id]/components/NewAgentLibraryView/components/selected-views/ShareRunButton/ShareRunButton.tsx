"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  CheckIcon,
  CopyIcon,
  ShareFatIcon,
  WarningIcon,
} from "@phosphor-icons/react";
import { useShareRunButton } from "./useShareRunButton";

interface Props {
  graphId: string;
  executionId: string;
  isShared?: boolean;
  shareToken?: string | null;
}

export function ShareRunButton({
  graphId,
  executionId,
  isShared: initialIsShared = false,
  shareToken: initialShareToken,
}: Props) {
  const {
    isShared,
    shareUrl,
    copied,
    loading,
    handleShare,
    handleStopSharing,
    handleCopy,
  } = useShareRunButton({
    graphId,
    executionId,
    isShared: initialIsShared,
    shareToken: initialShareToken,
  });

  return (
    <Dialog
      title="Share Agent Run"
      styling={{ maxWidth: "36rem", minWidth: "auto" }}
    >
      <Dialog.Trigger>
        <Button
          variant="icon"
          size="icon"
          aria-label="Share results"
          className={isShared ? "relative" : ""}
        >
          <ShareFatIcon weight="bold" size={18} className="text-zinc-700" />
        </Button>
      </Dialog.Trigger>

      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="large">
            {isShared
              ? "Your agent run is currently shared. Anyone with the link can view the output."
              : "Generate a public link to share this agent run output with others."}
          </Text>

          {!isShared ? (
            <>
              <div className="!mb-4">
                <Alert>
                  <WarningIcon className="h-4 w-4" />
                  <Text variant="body">
                    When you enable sharing, the output of this agent run will
                    be publicly accessible to anyone with the link. The page
                    will include a noindex directive to discourage search engine
                    crawling, but this cannot be guaranteed.
                  </Text>
                </Alert>
              </div>
              <Button
                onClick={handleShare}
                loading={loading}
                className="mt-6 w-full"
              >
                Enable Sharing
              </Button>
            </>
          ) : (
            <>
              <div className="flex w-full items-center gap-4">
                <Input
                  type="text"
                  value={shareUrl}
                  readOnly
                  label="Share URL"
                  id="share-url"
                  size="small"
                  className="!m-0"
                  wrapperClassName="flex-1"
                />
                <Button
                  variant="secondary"
                  onClick={handleCopy}
                  size="small"
                  className="mt-0.5 !min-w-0"
                >
                  {copied ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                </Button>
              </div>
              <Alert>
                <WarningIcon className="h-4 w-4" />
                <AlertDescription>
                  This link is publicly accessible. Only share it with people
                  you trust. The shared page includes noindex directives to
                  discourage search engines.
                </AlertDescription>
              </Alert>
              <Button
                onClick={handleStopSharing}
                loading={loading}
                variant="destructive"
                className="mt-6 w-full"
              >
                Stop Sharing
              </Button>
            </>
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

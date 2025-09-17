"use client";

import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  ShareFatIcon,
  CopyIcon,
  CheckIcon,
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
    <Dialog title="Share Agent Run">
      <Dialog.Trigger>
        <Button
          variant={isShared ? "primary" : "secondary"}
          size="small"
          className={isShared ? "relative" : ""}
        >
          <ShareFatIcon size={16} />
          {isShared ? "Shared" : "Share"}
          {isShared && (
            <span className="absolute -right-1 -top-1 h-2 w-2 rounded-full bg-green-500" />
          )}
        </Button>
      </Dialog.Trigger>

      <Dialog.Content>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            {isShared
              ? "Your agent run is currently shared. Anyone with the link can view the output."
              : "Generate a public link to share this agent run output with others."}
          </p>

          {!isShared ? (
            <>
              <Alert>
                <WarningIcon className="h-4 w-4" />
                <AlertDescription>
                  When you enable sharing, the output of this agent run will be
                  publicly accessible to anyone with the link. The page will
                  include a noindex directive to discourage search engine
                  crawling, but this cannot be guaranteed.
                </AlertDescription>
              </Alert>
              <Button
                onClick={handleShare}
                loading={loading}
                className="w-full"
              >
                Enable Sharing
              </Button>
            </>
          ) : (
            <>
              <div className="flex items-center space-x-2">
                <Input value={shareUrl} readOnly className="flex-1" />
                <button
                  onClick={handleCopy}
                  className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-input bg-background text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
                  type="button"
                >
                  {copied ? <CheckIcon size={16} /> : <CopyIcon size={16} />}
                </button>
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
                className="w-full"
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

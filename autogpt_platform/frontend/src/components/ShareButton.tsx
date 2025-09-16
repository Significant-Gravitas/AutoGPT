"use client";

import React, { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ShareFat, Copy, Check, Warning } from "@phosphor-icons/react";
import { useToast } from "./molecules/Toast/use-toast";
import {
  usePostV1EnableExecutionSharing,
  useDeleteV1DisableExecutionSharing,
} from "@/app/api/__generated__/endpoints/default/default";

interface ShareButtonProps {
  graphId: string;
  executionId: string;
  isShared?: boolean;
  shareUrl?: string;
}

export function ShareButton({
  graphId,
  executionId,
  isShared: initialIsShared = false,
  shareUrl: initialShareUrl,
}: ShareButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isShared, setIsShared] = useState(initialIsShared);
  const [shareUrl, setShareUrl] = useState(initialShareUrl || "");
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  const { mutate: enableSharing, isPending: isEnabling } =
    usePostV1EnableExecutionSharing();
  const { mutate: disableSharing, isPending: isDisabling } =
    useDeleteV1DisableExecutionSharing();

  const loading = isEnabling || isDisabling;

  const handleShare = () => {
    enableSharing(
      {
        graphId,
        graphExecId: executionId,
      },
      {
        onSuccess: (response) => {
          if (response.data) {
            if (response.status !== 200) {
              toast({
                title: "Error",
                description: "Failed to enable sharing. Please try again.",
                variant: "destructive",
              });
              return;
            }
            setShareUrl(response.data.share_url);
            setIsShared(true);
            toast({
              title: "Sharing enabled",
              description:
                "Your agent run is now publicly accessible via the share link.",
            });
          }
        },
        onError: () => {
          toast({
            title: "Error",
            description: "Failed to enable sharing. Please try again.",
            variant: "destructive",
          });
        },
      },
    );
  };

  const handleStopSharing = () => {
    disableSharing(
      {
        graphId,
        graphExecId: executionId,
      },
      {
        onSuccess: () => {
          setIsShared(false);
          setShareUrl("");
          toast({
            title: "Sharing disabled",
            description: "The share link is no longer accessible.",
          });
        },
        onError: () => {
          toast({
            title: "Error",
            description: "Failed to disable sharing. Please try again.",
            variant: "destructive",
          });
        },
      },
    );
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast({
        title: "Copied!",
        description: "Share link copied to clipboard.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to copy link. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="secondary" size="small">
          <ShareFat size={16} />
          Share
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Share Agent Run</DialogTitle>
          <DialogDescription>
            {isShared
              ? "Your agent run is currently shared. Anyone with the link can view the output."
              : "Generate a public link to share this agent run output with others."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {!isShared ? (
            <>
              <Alert>
                <Warning className="h-4 w-4" />
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
                  {copied ? <Check size={16} /> : <Copy size={16} />}
                </button>
              </div>
              <Alert>
                <Warning className="h-4 w-4" />
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
      </DialogContent>
    </Dialog>
  );
}

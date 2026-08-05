"use client";

import { useState } from "react";
import { CheckIcon, LinkIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";

interface Props {
  url: string;
}

export function ShareLinkButton({ url }: Props) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    const shareUrl =
      typeof window !== "undefined" ? `${window.location.origin}${url}` : url;
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      toast({ title: "Link copied to clipboard" });
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast({
        title: "Couldn't copy link",
        variant: "destructive",
      });
    }
  }

  return (
    <Button
      variant="ghost"
      size="small"
      onClick={handleCopy}
      leftIcon={
        copied ? (
          <CheckIcon size={14} weight="bold" />
        ) : (
          <LinkIcon size={14} weight="bold" />
        )
      }
      className="w-full sm:w-auto"
      data-testid="copy-share-link-button"
    >
      {copied ? "Copied" : "Copy share link"}
    </Button>
  );
}

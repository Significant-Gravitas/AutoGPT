"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  LinkedinLogo,
  RedditLogo,
  LinkSimple,
  XLogo,
} from "@phosphor-icons/react";
import { useRouter } from "next/navigation";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const PLATFORMS = [
  {
    name: "X (Twitter)",
    icon: XLogo,
    prompt:
      "Help me create an automation that shares a post about AutoGPT on X/Twitter. I'd like to share how I've been using the platform to automate my tasks.",
  },
  {
    name: "LinkedIn",
    icon: LinkedinLogo,
    prompt:
      "Help me create an automation that shares a post about AutoGPT on LinkedIn. I'd like to share how I've been using the platform to automate my work.",
  },
  {
    name: "Reddit",
    icon: RedditLogo,
    prompt:
      "Help me create an automation that shares a post about AutoGPT on Reddit. I'd like to share my experience using the platform.",
  },
  {
    name: "Other",
    icon: LinkSimple,
    prompt:
      "Help me create an automation that shares a post about AutoGPT on social media. I'd like to share how I've been using the platform.",
  },
] as const;

export function SharePlatformDialog({ open, onOpenChange }: Props) {
  const router = useRouter();

  function handlePlatformClick(prompt: string) {
    onOpenChange(false);
    const encodedPrompt = encodeURIComponent(prompt);
    router.push(`/copilot#prompt=${encodedPrompt}`);
  }

  return (
    <Dialog
      title="Share AutoGPT"
      styling={{ maxWidth: "28rem", minWidth: "auto" }}
      controlled={{
        isOpen: open,
        set: async (isOpen) => onOpenChange(isOpen),
      }}
    >
      <Dialog.Content>
        <Text variant="body" className="mb-4">
          Pick a platform and the Copilot will help you create a sharing
          automation.
        </Text>
        <div className="flex flex-col gap-2">
          {PLATFORMS.map((platform) => (
            <Button
              key={platform.name}
              variant="secondary"
              className="flex w-full items-center justify-start gap-3 px-4 py-3"
              onClick={() => handlePlatformClick(platform.prompt)}
            >
              <platform.icon size={20} />
              <span>{platform.name}</span>
            </Button>
          ))}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

"use client";

import { CopyIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { toast } from "@/components/molecules/Toast/use-toast";

interface Props {
  plainTextKey: string;
  onClose: () => void;
}

export function CreateAPIKeySuccess({ plainTextKey, onClose }: Props) {
  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(plainTextKey);
      toast({ title: "Copied to clipboard", variant: "success" });
    } catch {
      toast({
        title: "Could not copy to clipboard",
        description: "Please copy the key manually.",
        variant: "destructive",
      });
    }
  }

  return (
    <div className="flex flex-col gap-4 px-1">
      <Text variant="body" className="text-zinc-700">
        Copy your key now. For security, we won&apos;t show it again.
      </Text>

      <div className="flex items-center gap-2 rounded-md border border-zinc-200 bg-zinc-50 p-3">
        <code className="flex-1 break-all font-mono text-xs text-zinc-800">
          {plainTextKey}
        </code>
        <Button
          variant="secondary"
          size="small"
          leftIcon={<CopyIcon size={16} />}
          onClick={handleCopy}
        >
          Copy
        </Button>
      </div>

      <Button variant="primary" size="large" onClick={onClose}>
        Close
      </Button>
    </div>
  );
}

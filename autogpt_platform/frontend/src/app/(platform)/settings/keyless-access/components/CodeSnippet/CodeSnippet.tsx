"use client";

import { Copy01Icon } from "@hugeicons/core-free-icons";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { toast } from "@/components/molecules/Toast/use-toast";

interface Props {
  label: string;
  code: string;
}

export function CodeSnippet({ label, code }: Props) {
  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(code);
      toast({ title: "Copied to clipboard", variant: "success" });
    } catch {
      toast({
        title: "Could not copy to clipboard",
        description: "Please select the text and copy it manually.",
        variant: "destructive",
      });
    }
  }

  return (
    <div className="overflow-hidden rounded-md border border-zinc-200 bg-zinc-50">
      <div className="flex items-center justify-between border-b border-zinc-200 py-1 pl-3 pr-1">
        <Text variant="small" as="span" className="text-zinc-600">
          {label}
        </Text>
        <Button
          variant="ghost"
          size="small"
          aria-label={`Copy ${label}`}
          leftIcon={<Icon icon={Copy01Icon} size={14} />}
          onClick={handleCopy}
          className="!min-w-0"
        >
          Copy
        </Button>
      </div>
      <pre className="overflow-x-auto p-3 font-mono text-xs leading-5 text-zinc-800">
        <code>{code}</code>
      </pre>
    </div>
  );
}

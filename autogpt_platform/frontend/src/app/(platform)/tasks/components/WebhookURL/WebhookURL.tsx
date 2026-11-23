"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";

interface Props {
  url: string;
}

/** The URL a watching task listens on, for the user to give the service
 *  that sends the events, with a button that copies it. */
export function WebhookURL({ url }: Props) {
  const [isCopied, setIsCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(url);
      setIsCopied(true);
    } catch {
      // No clipboard (a page not served over HTTPS): the URL stays readable.
    }
  }

  return (
    <span className="ph-no-capture flex min-w-0 items-center gap-1">
      <Text
        as="code"
        variant="small-mono"
        tone="quiet"
        className="min-w-0 flex-1 break-all"
      >
        {url}
      </Text>
      <Button
        variant="ghost-muted"
        size="icon-xs"
        aria-label={isCopied ? "Copied" : "Copy webhook URL"}
        withTooltip={false}
        leadingIcon={isCopied ? Tick02Icon : Copy01Icon}
        onClick={handleCopy}
      />
    </span>
  );
}

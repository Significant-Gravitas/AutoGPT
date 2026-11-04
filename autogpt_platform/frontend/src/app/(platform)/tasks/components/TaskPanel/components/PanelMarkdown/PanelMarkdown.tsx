"use client";

import { Text } from "@/components/atoms/Text/Text";
import type { Variant } from "@/components/atoms/Text/helpers";
import { MessageResponse } from "@/components/organisms/Message/Message";
import type { Tone } from "@/components/styles/tones";

interface Props {
  children: string;
  /** One line of text: paragraphs run inline, so `**bold**` still renders
   *  inside a feed line. */
  inline?: boolean;
  variant?: Variant;
  tone?: Tone;
}

/** Text written by Otto or an expert, rendered as the chat renders it. The
 *  size and colour come from the Text around it. */
export function PanelMarkdown({
  children,
  inline = false,
  variant = "inbox",
  tone = "ink",
}: Props) {
  return (
    <Text as="div" variant={variant} tone={tone} className="min-w-0 flex-1">
      {inline ? (
        <MessageResponse className="size-auto break-words [&_p]:my-0 [&_p]:inline">
          {children}
        </MessageResponse>
      ) : (
        <MessageResponse className="size-auto break-words [&_li]:my-0.5 [&_ol]:my-1.5 [&_p]:my-0 [&_p+p]:mt-2 [&_ul]:my-1.5">
          {children}
        </MessageResponse>
      )}
    </Text>
  );
}

import type { ComponentRenderProps } from "@openuidev/react-lang";
import { defaultRehypePlugins } from "streamdown";
import type { z } from "zod/v4";
import { MessageResponse } from "@/components/ai-elements/message";
import type { Prose } from "@/lib/openui/catalog";

// Without rehype-raw, HTML in the model's Markdown is dropped instead of rendered.
const rehypePlugins = [
  defaultRehypePlugins.sanitize,
  defaultRehypePlugins.harden,
];

export function ProseView({
  props,
}: ComponentRenderProps<z.infer<typeof Prose.props>>) {
  return (
    <MessageResponse
      className="text-sm leading-relaxed text-zinc-700"
      rehypePlugins={rehypePlugins}
    >
      {props.markdown ?? ""}
    </MessageResponse>
  );
}

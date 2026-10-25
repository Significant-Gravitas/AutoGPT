import { createElement } from "react";
import {
  MessageResponse,
  type MessageResponseProps,
} from "@/components/organisms/Message/Message";
import {
  createComponentsMerger,
  type StreamdownComponents,
} from "@/components/organisms/Message/streamdown-components";
import { useReadEveryRender } from "@/hooks/useReadEveryRender";
import { CredentialMentionBadge } from "./CredentialMentionBadge";
import {
  credentialMentionImageProvider,
  credentialMentionsToMarkdown,
} from "./helpers";

interface Props extends Omit<MessageResponseProps, "children"> {
  children: string;
}

const NO_COMPONENTS: StreamdownComponents = {};

function createImageRenderer(Image: StreamdownComponents["img"]) {
  return function renderImage(imageProps: React.JSX.IntrinsicElements["img"]) {
    const provider = credentialMentionImageProvider(
      typeof imageProps.src === "string" ? imageProps.src : undefined,
    );
    if (provider !== null)
      return (
        <CredentialMentionBadge
          provider={provider}
          name={imageProps.alt ?? "Account"}
        />
      );
    if (typeof Image === "string") return createElement(Image, imageProps);
    return Image ? <Image {...imageProps} /> : null;
  };
}

// A new `img` per render would make Streamdown re-parse every block and remount
// every image on each streamed chunk, so it is built once per `components`.
const withCredentialBadges = createComponentsMerger((components) => ({
  ...components,
  img: createImageRenderer(components.img),
}));

export function CredentialMentionMarkdown({
  children,
  components,
  ...props
}: Props) {
  // Merged on every render, as a plain call would be: the merger keeps the
  // result stable and spots an `img` the caller replaced in place.
  const mergedComponents = useReadEveryRender(() =>
    withCredentialBadges(components ?? NO_COMPONENTS),
  );
  return (
    <MessageResponse {...props} components={mergedComponents}>
      {credentialMentionsToMarkdown(children)}
    </MessageResponse>
  );
}

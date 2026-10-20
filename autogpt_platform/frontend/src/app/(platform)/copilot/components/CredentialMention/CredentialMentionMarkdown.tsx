import { createElement } from "react";
import {
  MessageResponse,
  type MessageResponseProps,
} from "@/components/ai-elements/message";
import {
  createComponentsMerger,
  type StreamdownComponents,
} from "@/components/ai-elements/streamdown-components";
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
  return (
    <MessageResponse
      {...props}
      components={withCredentialBadges(components ?? NO_COMPONENTS)}
    >
      {credentialMentionsToMarkdown(children)}
    </MessageResponse>
  );
}

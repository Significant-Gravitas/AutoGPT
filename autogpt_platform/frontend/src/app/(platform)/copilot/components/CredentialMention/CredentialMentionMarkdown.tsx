import { createElement } from "react";
import {
  MessageResponse,
  type MessageResponseProps,
} from "@/components/ai-elements/message";
import { CredentialMentionBadge } from "./CredentialMentionBadge";
import {
  credentialMentionImageProvider,
  credentialMentionsToMarkdown,
} from "./helpers";

interface Props extends Omit<MessageResponseProps, "children"> {
  children: string;
}

type StreamdownComponents = NonNullable<MessageResponseProps["components"]>;

const NO_COMPONENTS: StreamdownComponents = {};

// A new `img` per render would make Streamdown re-parse every block and remount
// every image on each streamed chunk, so it is built once per `components`.
const componentsWithBadges = new WeakMap<
  StreamdownComponents,
  StreamdownComponents
>();

function createImageRenderer(Image: StreamdownComponents["img"]) {
  return function renderImage(imageProps: React.JSX.IntrinsicElements["img"]) {
    const provider = credentialMentionImageProvider(imageProps.src);
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

function withCredentialBadges(components = NO_COMPONENTS) {
  let merged = componentsWithBadges.get(components);
  if (!merged) {
    merged = { ...components, img: createImageRenderer(components.img) };
    componentsWithBadges.set(components, merged);
  }
  return merged;
}

export function CredentialMentionMarkdown({
  children,
  components,
  ...props
}: Props) {
  return (
    <MessageResponse {...props} components={withCredentialBadges(components)}>
      {credentialMentionsToMarkdown(children)}
    </MessageResponse>
  );
}

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

export function CredentialMentionMarkdown({
  children,
  components,
  ...props
}: Props) {
  const Image = components?.img;
  function renderImage(imageProps: React.JSX.IntrinsicElements["img"]) {
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
  }
  return (
    <MessageResponse
      {...props}
      components={{ ...components, img: renderImage }}
    >
      {credentialMentionsToMarkdown(children)}
    </MessageResponse>
  );
}

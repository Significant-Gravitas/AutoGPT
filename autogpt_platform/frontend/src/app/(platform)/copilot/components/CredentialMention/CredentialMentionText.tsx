import { Fragment } from "react";
import { CredentialMentionBadge } from "./CredentialMentionBadge";
import { parseCredentialMentions } from "./helpers";

interface Props {
  text: string;
}

export function CredentialMentionText({ text }: Props) {
  return (
    <>
      {parseCredentialMentions(text).map((part, index) => (
        <Fragment key={index}>
          {typeof part === "string" ? (
            part
          ) : (
            <CredentialMentionBadge name={part.name} provider={part.provider} />
          )}
        </Fragment>
      ))}
    </>
  );
}

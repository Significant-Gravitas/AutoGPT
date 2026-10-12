export interface CredentialMention {
  credentialId: string;
  provider: string;
  name: string;
}

export interface CredentialMentionPart extends CredentialMention {
  token: string;
}

export const CREDENTIAL_IMAGE_PREFIX = "/__credential-mention/";
const MENTION_PATTERN =
  /\[((?:\\.|[^\\\]])*)\]\(credential:\/\/([^/\s]+)\/([^\s)]+)\)/g;

export function serializeCredentialMention(mention: CredentialMention) {
  const name = mention.name.replace(/[\\[\]]/g, "\\$&").replace(/\r?\n/g, " ");
  return `[${name}](credential://${encodeURIComponent(mention.provider)}/${encodeURIComponent(mention.credentialId)})`;
}

export function parseCredentialMentions(
  text: string,
): (string | CredentialMentionPart)[] {
  const parts: (string | CredentialMentionPart)[] = [];
  let end = 0;
  for (const match of text.matchAll(MENTION_PATTERN)) {
    let mention: CredentialMentionPart;
    try {
      mention = {
        name: match[1].replace(/\\(.)/g, "$1"),
        provider: decodeURIComponent(match[2]),
        credentialId: decodeURIComponent(match[3]),
        token: match[0],
      };
    } catch {
      continue;
    }
    if (match.index > end) parts.push(text.slice(end, match.index));
    parts.push(mention);
    end = match.index + match[0].length;
  }
  if (end < text.length) parts.push(text.slice(end));
  return parts;
}

export function credentialMentionDisplayText(text: string) {
  return parseCredentialMentions(text)
    .map((part) => (typeof part === "string" ? part : part.name))
    .join("");
}

export function credentialMentionsToMarkdown(text: string) {
  return parseCredentialMentions(text)
    .map((part) =>
      typeof part === "string"
        ? part
        : `![${part.name.replace(/[\\[\]]/g, "\\$&")}](${CREDENTIAL_IMAGE_PREFIX}${encodeURIComponent(part.provider)}/${encodeURIComponent(part.credentialId)})`,
    )
    .join("");
}

export function credentialMentionImageProvider(src?: string) {
  if (!src?.startsWith(CREDENTIAL_IMAGE_PREFIX)) return null;
  try {
    return decodeURIComponent(
      src.slice(CREDENTIAL_IMAGE_PREFIX.length).split("/")[0],
    );
  } catch {
    return null;
  }
}

// Paths of the image tokens the chat draws itself through its `img` override.
// Lowercase words and dashes only: Streamdown closes an odd `_` or `*` in the text.
export const CHAT_IMAGE_TOKEN_PREFIXES = {
  credentialMention: "/chat-token/credential-mention/",
  workspaceLink: "/chat-token/workspace-link/",
} as const;

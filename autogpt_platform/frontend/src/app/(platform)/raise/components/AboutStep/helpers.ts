const ABOUT_PLACEHOLDER =
  "How they should work, what you care about, anything that helps them sound like yours…";

export function aboutPlaceholderFor(name: string | null) {
  const trimmed = name?.trim();
  if (!trimmed) return ABOUT_PLACEHOLDER;
  return `How ${trimmed} should work, what you care about, anything that helps them sound like yours…`;
}

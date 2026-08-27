// Up-to-two-letter initials from an org name for the avatar fallback: first
// letters of the first two words, or the first two letters of a single word.
export function getOrgInitials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[1][0]).toUpperCase();
}

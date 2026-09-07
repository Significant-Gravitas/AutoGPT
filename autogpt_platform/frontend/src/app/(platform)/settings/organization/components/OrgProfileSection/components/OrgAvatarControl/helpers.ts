const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });

export function getOrgInitials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1)
    return graphemes(words[0]).slice(0, 2).join("").toUpperCase();
  return (graphemes(words[0])[0] + graphemes(words[1])[0]).toUpperCase();
}

function graphemes(value: string): string[] {
  return Array.from(segmenter.segment(value), ({ segment }) => segment);
}

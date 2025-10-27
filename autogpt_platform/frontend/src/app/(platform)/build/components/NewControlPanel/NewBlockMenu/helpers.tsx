export const highlightText = (
  text: string | undefined,
  highlight: string | undefined,
) => {
  if (!text || !highlight) return text;

  function escapeRegExp(s: string) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  const escaped = escapeRegExp(highlight);
  const parts = text.split(new RegExp(`(${escaped})`, "gi"));
  return parts.map((part, i) =>
    part.toLowerCase() === highlight?.toLowerCase() ? (
      <mark key={i} className="bg-transparent font-bold">
        {part}
      </mark>
    ) : (
      part
    ),
  );
};

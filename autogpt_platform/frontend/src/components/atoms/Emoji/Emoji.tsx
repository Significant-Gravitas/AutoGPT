import twemoji from "twemoji";

// Served from public/emoji so the browser never fetches a third-party CDN,
// which an offline or self-hosted install may not reach. Add an emoji's SVG
// there (Twemoji 14.0.2, named by codepoint) before using it; any other emoji
// renders as the system glyph.
const BUNDLED_EMOJI = new Set([
  "1f4a4",
  "1f4b5",
  "1f4c5",
  "1f5c3",
  "1f6a9",
  "26a0",
]);

function bundledEmojiURL(icon: string) {
  return BUNDLED_EMOJI.has(icon) ? `/emoji/${icon}.svg` : false;
}

interface Props {
  text: string;
  size?: number;
}

// twemoji.parse passes non-emoji characters through verbatim and the official
// docs explicitly disclaim string sanitisation, so feeding raw `text` and
// dumping the result through dangerouslySetInnerHTML would be XSS-able if a
// caller ever passed user input. Escaping HTML metacharacters first keeps
// emoji codepoints (not metacharacters) untouched while neutering any
// embedded markup.
function escapeHtml(input: string) {
  return input
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function Emoji({ text, size = 24 }: Props) {
  // Strip the default alt (the emoji codepoint) so the browser doesn't flash
  // the small system-rendered emoji while the SVG loads. The
  // twemoji API doesn't expose an alt override, hence the regex.
  const html = twemoji
    .parse(escapeHtml(text), {
      callback: bundledEmojiURL,
      attributes: () => ({
        width: String(size),
        height: String(size),
      }),
    })
    .replace(/\salt="[^"]*"/, ' alt=""');

  return (
    <span
      className="inline-flex items-center"
      style={{ width: size, height: size }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

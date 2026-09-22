import twemoji from "twemoji";

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
  // the small system-rendered emoji while the SVG loads from the CDN. The
  // twemoji API doesn't expose an alt override, hence the regex.
  const html = twemoji
    .parse(escapeHtml(text), {
      folder: "svg",
      ext: ".svg",
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

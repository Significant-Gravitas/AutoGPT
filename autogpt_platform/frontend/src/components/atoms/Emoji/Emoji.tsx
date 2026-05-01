import twemoji from "twemoji";

interface Props {
  text: string;
  size?: number;
}

export function Emoji({ text, size = 24 }: Props) {
  // twemoji.parse only converts emoji codepoints to <img> tags pointing to
  // Twitter's CDN SVGs — it does not inject arbitrary HTML.
  //
  // Strip the default alt (the emoji codepoint) so the browser doesn't flash
  // the small system-rendered emoji while the SVG loads from the CDN. The
  // twemoji API doesn't expose an alt override, hence the regex.
  const html = twemoji
    .parse(text, {
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

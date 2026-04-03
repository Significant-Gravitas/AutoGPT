import twemoji from "twemoji";

interface Props {
  text: string;
  size?: number;
}

export function Emoji({ text, size = 24 }: Props) {
  return (
    <span
      className="inline-flex items-center"
      style={{ width: size, height: size }}
      dangerouslySetInnerHTML={{
        // twemoji.parse only converts emoji codepoints to <img> tags
        // pointing to Twitter's CDN SVGs — it does not inject arbitrary HTML
        __html: twemoji.parse(text, {
          folder: "svg",
          ext: ".svg",
          attributes: () => ({
            width: String(size),
            height: String(size),
          }),
        }),
      }}
    />
  );
}

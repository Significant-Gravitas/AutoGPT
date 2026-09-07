import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import { parseAvatarUrl } from "@/components/molecules/BotAvatar/helpers";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

export const dynamic = "force-static";

const SVG_XMLNS = 'xmlns="http://www.w3.org/2000/svg"';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ file: string }> },
) {
  const { file } = await params;
  const config = parseAvatarUrl(`/avatars/${file}`);
  if (!config) return new Response("Not found", { status: 404 });

  const markup = renderToStaticMarkup(
    createElement(BotAvatar, { config, animated: false, size: 512 }),
  ).replace("<svg ", `<svg ${SVG_XMLNS} `);

  return new Response(markup, {
    headers: {
      "Content-Type": "image/svg+xml; charset=utf-8",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}

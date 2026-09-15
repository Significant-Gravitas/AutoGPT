import { BotAvatarSvg } from "@/components/molecules/BotAvatar/BotAvatarSvg";
import { parseAvatarUrl } from "@/components/molecules/BotAvatar/helpers";
import { FRONT_POSE } from "@/components/molecules/BotAvatar/projection";
import { STATIC_ELS } from "@/components/molecules/BotAvatar/svgElements";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server.browser";

export const dynamic = "force-static";

const SVG_XMLNS = 'xmlns="http://www.w3.org/2000/svg"';

// Route handlers run in the server layer, where hooks and framer-motion's
// client components are unavailable — so this renders the avatar at rest
// through the plain-element path rather than BotAvatar itself.
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ file: string }> },
) {
  const { file } = await params;
  const config = parseAvatarUrl(`/avatars/${file}`);
  if (!config) return new Response("Not found", { status: 404 });

  const markup = renderToStaticMarkup(
    createElement(BotAvatarSvg, {
      config,
      status: "idle",
      expression: "neutral",
      pose: FRONT_POSE,
      isLive: false,
      isBlinking: false,
      els: STATIC_ELS,
      idPrefix: "avatar",
      size: 512,
    }),
  ).replace("<svg ", `<svg ${SVG_XMLNS} `);

  return new Response(markup, {
    headers: {
      "Content-Type": "image/svg+xml; charset=utf-8",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}

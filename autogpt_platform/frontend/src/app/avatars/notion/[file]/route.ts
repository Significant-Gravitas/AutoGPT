import { composeNotionAvatar } from "@/components/molecules/NotionAvatar/compose";
import { decodeNotionConfig } from "@/components/molecules/NotionAvatar/helpers";

// Every slot combination is a valid URL, so there is nothing to prerender.
// The markup is deterministic, hence the immutable cache.
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ file: string }> },
) {
  const { file } = await params;
  if (!file.endsWith(".svg")) return new Response("Not found", { status: 404 });

  const config = decodeNotionConfig(file.slice(0, -".svg".length));
  if (!config) return new Response("Not found", { status: 404 });

  return new Response(composeNotionAvatar(config, { size: 512 }), {
    headers: {
      "Content-Type": "image/svg+xml; charset=utf-8",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}

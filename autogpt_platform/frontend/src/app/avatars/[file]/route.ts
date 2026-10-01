import {
  notionAvatarImageUrlFor,
  notionConfigForLegacyUrl,
} from "@/components/molecules/NotionAvatar/helpers";

export const dynamic = "force-dynamic";

// Experts raised before the Notion artwork landed still hold URLs of the form
// /avatars/<shape>.<color>.<accessory>.svg. Rather than backfill the column,
// the old shape maps onto a face and redirects — permanently, so the browser
// and any CDN stop asking.
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ file: string }> },
) {
  const { file } = await params;
  const config = notionConfigForLegacyUrl(`/avatars/${file}`);
  if (!config) return new Response("Not found", { status: 404 });

  return Response.redirect(
    new URL(notionAvatarImageUrlFor(config), _request.url),
    308,
  );
}

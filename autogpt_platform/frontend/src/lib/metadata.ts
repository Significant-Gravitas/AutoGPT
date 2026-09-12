import type { Metadata } from "next";

const SITE_NAME = "AutoGPT";
// X truncates around 200 characters, the tightest of the platforms we target.
const DESCRIPTION_MAX_LENGTH = 200;

interface PageMetadataOptions {
  title: string;
  description?: string | null;
  path?: string;
  images?: (string | null | undefined)[];
  type?: "website" | "article" | "profile";
}

// Emits og:image only for a real image; unfurlers render a broken one worse
// than none at all.
export function buildPageMetadata({
  title,
  description,
  path,
  images,
  type = "website",
}: PageMetadataOptions): Metadata {
  const url = path ? new URL(path, getSiteUrl()).toString() : undefined;
  const cardImages = (images ?? []).filter(
    (image): image is string => typeof image === "string" && image.length > 0,
  );
  const summary = toCardDescription(description);

  return {
    title,
    description: summary,
    alternates: url ? { canonical: url } : undefined,
    openGraph: {
      title,
      description: summary,
      siteName: SITE_NAME,
      type,
      url,
      ...(cardImages.length > 0 ? { images: cardImages } : {}),
    },
    twitter: {
      card: cardImages.length > 0 ? "summary_large_image" : "summary",
      title,
      description: summary,
      ...(cardImages.length > 0 ? { images: cardImages } : {}),
    },
  };
}

// Unfurlers cut a long description at their own limit, mid-word; cutting on a
// word boundary first keeps the last word of the card whole.
function toCardDescription(description?: string | null): string | undefined {
  const text = description?.trim();
  if (!text) return undefined;
  if (text.length <= DESCRIPTION_MAX_LENGTH) return text;

  const clipped = text.slice(0, DESCRIPTION_MAX_LENGTH);
  const lastSpace = clipped.lastIndexOf(" ");
  const body = lastSpace > 0 ? clipped.slice(0, lastSpace) : clipped;
  return `${body.trimEnd()}…`;
}

// Falls back rather than returning a value `new URL()` would reject: the root
// layout builds metadataBase from this, so a bad origin here fails the build.
export function getSiteUrl(): string {
  const vercel = process.env.VERCEL_URL;

  return (
    firstValidOrigin([
      process.env.NEXT_PUBLIC_FRONTEND_BASE_URL,
      vercel ? `https://${vercel}` : undefined,
    ]) ?? "http://localhost:3000"
  );
}

function firstValidOrigin(
  candidates: (string | undefined)[],
): string | undefined {
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      const url = new URL(candidate);
      // mailto: and data: parse but have no origin, so resolving a relative
      // path against one throws instead of falling through to the next.
      if (url.protocol !== "http:" && url.protocol !== "https:") continue;
      return url.toString().replace(/\/$/, "");
    } catch {
      continue;
    }
  }
  return undefined;
}

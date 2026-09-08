import type { Metadata } from "next";

const SITE_NAME = "AutoGPT";

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
  const summary = description || undefined;

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
      return new URL(candidate).toString().replace(/\/$/, "");
    } catch {
      continue;
    }
  }
  return undefined;
}

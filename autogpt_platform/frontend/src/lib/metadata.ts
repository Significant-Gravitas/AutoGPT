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

export function getSiteUrl(): string {
  const configured = process.env.NEXT_PUBLIC_FRONTEND_BASE_URL;
  if (configured) return configured;

  // Next falls back to VERCEL_URL when metadataBase is unset, which is a
  // per-deployment hostname; making it explicit keeps the two in step.
  const vercel = process.env.VERCEL_URL;
  if (vercel) return `https://${vercel}`;

  return "http://localhost:3000";
}

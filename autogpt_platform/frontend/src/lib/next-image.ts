/** The hosts `next.config.mjs` lets `next/image` load from.
 *
 *  `next/image` throws while rendering — it never reaches `onError` — when
 *  `src` points at any other host, so a URL that arrives from user-submitted
 *  data has to be checked before it is handed to an `<Image>`.
 *  `next-image.test.ts` keeps this list in step with the config. */
const ALLOWED_IMAGE_HOSTS = [
  "ddz4ak4pa3d19.cloudfront.net",
  "example.com",
  "ideogram.ai",
  "images.unsplash.com",
  "lh3.googleusercontent.com",
  "storage.cloud.google.com",
  "storage.googleapis.com",
  "upload.wikimedia.org",
];

/** Whether `next/image` can render this src instead of throwing on it. */
export function isRenderableImageUrl(
  url: string | null | undefined,
): url is string {
  if (!url) return false;
  if (url.startsWith("/")) return true;
  try {
    return ALLOWED_IMAGE_HOSTS.includes(new URL(url).hostname);
  } catch {
    return false;
  }
}

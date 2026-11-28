import fs from "fs";
import path from "path";

// The images public/'s own folders hold, for the route crawl's check that a
// page asks Next's image optimizer for none that isn't there.

const PUBLIC_DIR = path.resolve(__dirname, "../../../public");

/** The image an optimizer request (`/_next/image?url=…`) asks for, when it is
 *  under one of public/'s folders and public/ doesn't have it; else null. */
export function missingPublicImage(url: string) {
  const image = new URL(url).searchParams.get("url");
  if (!image || !inAPublicFolder(image) || existsInPublic(image)) return null;
  return image;
}

/**
 * A local path under one of public/'s folders (/integrations/…). Anything
 * else, such as a route handler's (/avatars/…), another host's image or a
 * path that can't be decoded, is none of the check's business.
 */
export function inAPublicFolder(image: string) {
  const [, folder, file] = image.split("/");
  return (
    image.startsWith("/") &&
    Boolean(folder && file) &&
    publicFileAt(image) !== null &&
    fs.existsSync(path.join(PUBLIC_DIR, folder))
  );
}

/** Whether public/ has the file the app serves at `image`. */
export function existsInPublic(image: string) {
  const file = publicFileAt(image);
  return file !== null && fs.existsSync(file);
}

// The file under public/ the app serves at `image`, or null for a path that
// leads out of public/ or has a malformed percent sequence (%E0%A4%A).
function publicFileAt(image: string) {
  let decoded: string;
  try {
    decoded = decodeURIComponent(image.split("?")[0]);
  } catch {
    return null;
  }
  const file = path.join(PUBLIC_DIR, decoded);
  return file.startsWith(PUBLIC_DIR + path.sep) ? file : null;
}

const LOCAL_STORE_IMAGE_PATH =
  /\/api\/store\/media\/(?!\.{1,2}\/)[A-Za-z0-9_.-]+\/images\/(?!\.{1,2}$)[A-Za-z0-9_.-]+$/;

export function isLocalStoreMediaUrl(src: string | null | undefined): boolean {
  if (!src || src.startsWith("//")) return false;

  const match = /^(?:https?:\/\/[^/?#]+)?(\/[^?#]*)(?:[?#].*)?$/i.exec(src);
  if (!match) return false;

  try {
    const frontendURL = new URL(getFrontendOrigin());
    const url = new URL(src, frontendURL);
    if (
      url.username ||
      url.password ||
      url.pathname !== match[1] ||
      !LOCAL_STORE_IMAGE_PATH.test(url.pathname)
    ) {
      return false;
    }

    const apiPath = url.pathname.slice(
      0,
      url.pathname.lastIndexOf("/store/media/"),
    );
    if (
      url.origin === frontendURL.origin &&
      (apiPath === "/api" || apiPath === "/api/proxy/api")
    ) {
      return true;
    }

    const backendURL = new URL(
      process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006/api",
      frontendURL,
    );
    return (
      url.origin === backendURL.origin &&
      apiPath === backendURL.pathname.replace(/\/$/, "")
    );
  } catch {
    return false;
  }
}

function getFrontendOrigin(): string {
  if (typeof window !== "undefined") return window.location.origin;

  return (
    process.env.BETTER_AUTH_URL ||
    process.env.NEXT_PUBLIC_FRONTEND_BASE_URL ||
    "http://localhost:3000"
  );
}

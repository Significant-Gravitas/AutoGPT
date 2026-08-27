export function resolveOrgAvatarUrl(url: string | null) {
  if (!url) return null;
  if (url.startsWith("/api/proxy/")) return url;
  return url.startsWith("/api/") ? `/api/proxy${url}` : url;
}

export function isProtectedOrgAvatarUrl(url: string | null) {
  return Boolean(url?.startsWith("/api/proxy/api/orgs/"));
}

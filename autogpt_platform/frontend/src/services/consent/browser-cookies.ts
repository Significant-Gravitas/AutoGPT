/** The names of the cookies this page can read, each once. */
export function readCookieNames(): string[] {
  if (typeof document === "undefined") return [];
  const names = document.cookie
    .split(";")
    .map((part) => part.trim().split("=")[0])
    .filter(Boolean);
  return Array.from(new Set(names));
}

/**
 * Expires `name` on this host and on every parent domain, path `/`: trackers
 * set their cookies on the widest domain they can (.agpt.co from
 * platform.agpt.co), and a page cannot tell which domain a cookie came from.
 */
export function deleteCookie(name: string) {
  if (typeof document === "undefined") return;
  const expired = `${name}=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;
  document.cookie = expired;
  const labels = window.location.hostname.split(".");
  for (let index = 0; index < labels.length - 1; index++) {
    document.cookie = `${expired}; Domain=.${labels.slice(index).join(".")}`;
  }
}

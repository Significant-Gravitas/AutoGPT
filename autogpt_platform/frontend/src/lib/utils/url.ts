/**
 * Extracts the host from a URL string.
 * @param url - The URL string to extract the host from
 * @returns The host, including a non-default port, if valid; null if invalid
 */
export const getHostFromUrl = (url: string): string | null => {
  try {
    if (!/^https?:\/\//i.test(url)) {
      url = "http://" + url; // Add a scheme if missing for URL parsing
    }
    const urlObj = new URL(url);
    // `host`, not `hostname`: the backend refuses a request port outside
    // 80/443 unless the credential's host names it. `URL` drops default ports.
    return urlObj.host;
  } catch {
    return null;
  }
};

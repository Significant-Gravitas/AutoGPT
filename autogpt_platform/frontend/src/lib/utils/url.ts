/**
 * Extracts the hostname from a URL string.
 * @param url - The URL string to extract the hostname from
 * @returns The hostname if valid, null if invalid
 */
export const getHostFromUrl = (url: string): string | null => {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname;
  } catch {
    return null;
  }
};

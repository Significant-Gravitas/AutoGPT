// The server and the hydrating browser can have different default locales.
const countFormatter = new Intl.NumberFormat("en-US");

export function formatMarketplaceCount(count: number): string {
  return countFormatter.format(count);
}

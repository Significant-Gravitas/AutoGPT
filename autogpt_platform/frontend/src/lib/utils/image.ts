export function shouldBypassImageOptimization(src: string): boolean {
  return src.startsWith("/");
}

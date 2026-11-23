export function maskAPIKey(head: string, tail: string): string {
  return `${head}••••••••${tail}`;
}

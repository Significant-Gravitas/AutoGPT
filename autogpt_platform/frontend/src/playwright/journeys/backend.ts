// The stack's backend, for the calls a browser never makes itself: webhook
// deliveries and reads with no session. Against a remote frontend, say where
// that stack's backend is.
export function backendURL(): string {
  const explicit = process.env.PLAYWRIGHT_BACKEND_URL;
  if (explicit) return explicit;
  const base = new URL(
    process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000",
  );
  if (!["localhost", "127.0.0.1"].includes(base.hostname)) {
    throw new Error(
      `PLAYWRIGHT_BASE_URL is ${base.origin}: set PLAYWRIGHT_BACKEND_URL to that stack's backend`,
    );
  }
  return "http://localhost:8006";
}

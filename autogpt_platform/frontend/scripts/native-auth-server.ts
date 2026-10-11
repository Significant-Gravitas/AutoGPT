import { createServer } from "node:http";
import { toNodeHandler } from "better-auth/node";

async function main() {
  if (!process.env.DATABASE_URL || !process.env.BETTER_AUTH_SECRET) {
    throw new Error(
      "The auth harness requires a test database and auth secret",
    );
  }
  let handler: ReturnType<typeof toNodeHandler> | undefined;
  const server = createServer((request, response) => {
    if (!handler) {
      response.writeHead(503).end();
      return;
    }
    Promise.resolve(handler(request, response)).catch(() => {
      if (!response.headersSent) response.writeHead(500);
      response.end();
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(
      Number(process.env.NATIVE_AUTH_PORT || "0"),
      "127.0.0.1",
      resolve,
    );
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("The auth harness did not bind a TCP port");
  }
  const origin = `http://127.0.0.1:${address.port}`;
  process.env.BETTER_AUTH_URL = origin;
  const { auth } = await import("../src/lib/auth/auth");
  handler = toNodeHandler(auth);
  process.stdout.write(`NATIVE_AUTH_READY ${origin}\n`);
  process.on("SIGTERM", () => server.close(() => process.exit(0)));
}

main().catch(() => {
  console.error("Native auth harness startup failed");
  process.exit(1);
});

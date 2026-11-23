// @vitest-environment node
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { Socket } from "node:net";
import { fileURLToPath } from "node:url";
import { expect, test } from "vitest";

type Handler = (req: unknown, res: unknown) => Promise<void>;

// Next's own code from the installed package, which patches/next@15.5.28.patch
// changes: the optimizer's fetch of a local source image, and the static file
// handler the router runs for it.
const require = createRequire(import.meta.url);
const { fetchInternalImage } = require("next/dist/server/image-optimizer") as {
  fetchInternalImage: (
    href: string,
    req: unknown,
    res: unknown,
    maximumResponseBody: number,
    handleRequest: Handler,
  ) => Promise<{ buffer: Buffer }>;
};
const { serveStatic } = require("next/dist/server/serve-static") as {
  serveStatic: (req: unknown, res: unknown, path: string) => Promise<void>;
};

const SOURCE = "/experts/transparent/otto.webp";
const MAX_BODY = 50_000_000;

test("an image request whose browser has gone still reads its source file", async () => {
  // A navigation cancelled the request, so its connection is already closed
  // when the optimizer starts reading the file it resizes. Unpatched, Next
  // ties that read to the closed connection and it never ends: every later
  // request for the same image URL then waits on it (#16987).
  const socket = new Socket();
  socket.destroy();
  const file = fileURLToPath(
    new URL(`../../../../public${SOURCE}`, import.meta.url),
  );

  const upstream = fetchInternalImage(
    SOURCE,
    { method: "GET", socket },
    {},
    MAX_BODY,
    (req, res) => serveStatic(req, res, file),
  );

  const { buffer } = await settlesWithin(upstream, 5000);
  expect(Buffer.compare(buffer, readFileSync(file))).toBe(0);
});

test("Next's ESM build builds the image fetch's response without the browser's socket too", () => {
  // The test above runs the CommonJS build that `next start` and the
  // standalone server load; the patch changes the ESM copy the same way.
  const source = readFileSync(
    require.resolve("next/dist/esm/server/image-optimizer.js"),
    "utf8",
  );
  const response = /new MockedResponse\(\{([^}]*)\}\)/.exec(source);
  expect(
    response?.[1],
    "the vercel/next.js#98168 backport's ESM hunk is missing from patches/next@15.5.28.patch",
  ).toContain("maximumResponseBody");
  expect(response?.[1]).not.toContain("socket");
});

async function settlesWithin<T>(promise: Promise<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () =>
        reject(
          new Error(
            `Next's internal image fetch did not settle in ${ms} ms: is the vercel/next.js#98168 backport in patches/next@15.5.28.patch applied?`,
          ),
        ),
      ms,
    );
  });
  try {
    return await Promise.race([promise, deadline]);
  } finally {
    clearTimeout(timer);
  }
}

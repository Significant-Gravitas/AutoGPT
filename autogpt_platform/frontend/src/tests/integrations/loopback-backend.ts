import { server } from "@/mocks/mock-server";
import { http, passthrough } from "msw";
import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import { onTestFinished } from "vitest";

type Answer = (
  request: IncomingMessage,
  response: ServerResponse,
  count: number,
) => void;

/**
 * A real HTTP server on a loopback port, for a test that needs Node's own
 * fetch to meet a real connection: one that is dropped, reset or never
 * answered. `answer` gets each request and how many have arrived so far. MSW
 * lets requests to it through, and it closes when the test finishes. The test
 * file needs `@vitest-environment node`, or fetch is happy-dom's.
 */
export async function startLoopbackBackend(answer: Answer) {
  const requests: string[] = [];
  const backend = createServer((request, response) => {
    requests.push(`${request.method} ${request.url}`);
    answer(request, response, requests.length);
  });
  await new Promise<void>((resolve) => backend.listen(0, "127.0.0.1", resolve));
  onTestFinished(
    () =>
      new Promise<void>((resolve) => {
        backend.closeAllConnections();
        backend.close(() => resolve());
      }),
  );
  const { port } = backend.address() as AddressInfo;
  const url = `http://127.0.0.1:${port}`;
  server.use(http.all(`${url}/*`, () => passthrough()));
  return { url, requests };
}

/**
 * Hangs up without answering, once the whole request has arrived. Node's
 * fetch then fails with "other side closed" (UND_ERR_SOCKET) after a FIN, or
 * with ECONNRESET after a reset.
 */
export function hangUp(
  request: IncomingMessage,
  how: "destroy" | "resetAndDestroy" = "destroy",
) {
  request.resume();
  request.on("end", () => request.socket[how]());
}

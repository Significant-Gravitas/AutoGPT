import fs from "fs";
import http from "http";
import type { AddressInfo } from "net";
import path from "path";
import { FIXTURES_DIR } from "./paths";

/**
 * Replays a recorded `.sse` body one frame every `paceMs`, so a streamed
 * reply arrives at a steady, repeatable cadence instead of all at once.
 * The harness points the copilot stream request here with `route.continue`.
 * The fixture and pace of each stream are registered in-process by `open()`,
 * which hands out a tag that is unique for the life of the server; the
 * request only carries that tag.
 */
export interface StreamServer {
  open(args: { fixture: string; paceMs: number }): { url: string; tag: string };
  waitForDone(tag: string, timeoutMs: number): Promise<void>;
  close(): Promise<void>;
}

export function splitFrames(body: string) {
  return body
    .split(/\n\n+/)
    .map((frame) => frame.trim())
    .filter(Boolean);
}

export function readFixture(name: string) {
  if (!/^[\w.-]+$/.test(name)) throw new Error(`Bad fixture name: ${name}`);
  return fs.readFileSync(path.join(FIXTURES_DIR, name), "utf8");
}

function corsHeaders(request: http.IncomingMessage) {
  return {
    "Access-Control-Allow-Origin": request.headers.origin ?? "*",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  };
}

export async function startStreamServer(): Promise<StreamServer> {
  const finished = new Set<string>();
  const waiters = new Map<string, (() => void)[]>();
  const timers = new Set<ReturnType<typeof setTimeout>>();
  const streams = new Map<string, { frames: string[]; paceMs: number }>();
  let opened = 0;

  function markDone(tag: string) {
    finished.add(tag);
    waiters.get(tag)?.forEach((resolve) => resolve());
    waiters.delete(tag);
  }

  const server = http.createServer((request, response) => {
    if (request.method === "OPTIONS") {
      response.writeHead(204, corsHeaders(request)).end();
      return;
    }
    const params = new URL(request.url ?? "/", "http://localhost").searchParams;
    const tag = params.get("tag") ?? "";
    const stream = streams.get(tag);
    if (!stream) {
      response.writeHead(404, corsHeaders(request)).end();
      return;
    }
    const { frames, paceMs } = stream;
    response.writeHead(200, {
      ...corsHeaders(request),
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "x-vercel-ai-ui-message-stream": "v1",
    });
    let index = 0;
    const timer = setInterval(() => {
      if (index >= frames.length) {
        clearInterval(timer);
        response.end();
        markDone(tag);
        return;
      }
      response.write(`${frames[index++]}\n\n`);
    }, paceMs);
    response.on("close", () => clearInterval(timer));
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;

  return {
    open({ fixture, paceMs }) {
      const tag = `stream-${++opened}`;
      streams.set(tag, { frames: splitFrames(readFixture(fixture)), paceMs });
      const query = new URLSearchParams({ tag });
      return { url: `http://127.0.0.1:${port}/stream?${query}`, tag };
    },
    waitForDone(tag, timeoutMs) {
      if (finished.has(tag)) return Promise.resolve();
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          timers.delete(timeout);
          reject(new Error(`Stream "${tag}" did not finish`));
        }, timeoutMs);
        timers.add(timeout);
        waiters.set(tag, [
          ...(waiters.get(tag) ?? []),
          () => {
            clearTimeout(timeout);
            timers.delete(timeout);
            resolve();
          },
        ]);
      });
    },
    close() {
      timers.forEach((timer) => clearTimeout(timer));
      server.closeAllConnections();
      return new Promise((resolve) => server.close(() => resolve()));
    },
  };
}

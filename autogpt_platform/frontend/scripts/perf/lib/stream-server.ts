import fs from "fs";
import http from "http";
import type { AddressInfo } from "net";
import path from "path";
import { FIXTURES_DIR } from "./paths";

/**
 * Replays a recorded `.sse` body one frame every `paceMs`, so a streamed
 * reply arrives at a steady, repeatable cadence instead of all at once.
 * The harness points the copilot stream request here with `route.continue`.
 */
export interface StreamServer {
  url(args: { fixture: string; paceMs: number; tag: string }): string;
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
    const paceMs = Number(params.get("pace") ?? 30);
    let frames: string[];
    try {
      frames = splitFrames(readFixture(params.get("fixture") ?? ""));
    } catch (error) {
      response.writeHead(500, corsHeaders(request)).end(String(error));
      return;
    }
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
    request.on("close", () => clearInterval(timer));
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;

  return {
    url({ fixture, paceMs, tag }) {
      const query = new URLSearchParams({ fixture, pace: String(paceMs), tag });
      return `http://127.0.0.1:${port}/stream?${query}`;
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

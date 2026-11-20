import { afterEach, describe, expect, it } from "vitest";
import { startStreamServer, type StreamServer } from "../lib/stream-server";

const FIXTURE = "reply-20-lines.sse";

describe("stream server", () => {
  let server: StreamServer | null = null;

  afterEach(async () => {
    await server?.close();
    server = null;
  });

  it("gives every stream its own tag for the life of the server", async () => {
    server = await startStreamServer();
    const first = server.open({ fixture: FIXTURE, paceMs: 30 });
    const second = server.open({ fixture: FIXTURE, paceMs: 30 });

    expect(first.tag).not.toBe(second.tag);
    expect(first.url).not.toBe(second.url);
  });

  it("does not treat a new stream as done because an earlier one finished", async () => {
    server = await startStreamServer();
    const { tag } = server.open({ fixture: FIXTURE, paceMs: 30 });
    const settled = await Promise.race([
      server.waitForDone(tag, 200).then(
        () => "resolved",
        () => "timed out",
      ),
      new Promise((resolve) => setTimeout(() => resolve("pending"), 100)),
    ]);

    expect(settled).toBe("pending");
  });

  it("rejects an unknown fixture name", async () => {
    server = await startStreamServer();
    expect(() =>
      server!.open({ fixture: "../secrets.sse", paceMs: 30 }),
    ).toThrow("Bad fixture name");
  });
});

import { waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { saveBlob } from "../save-blob";
import { installDownloadBridge } from "./save-blob-fixtures";

afterEach(() => {
  delete window.AutoGPTDownloads;
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("Android blob download bridge", () => {
  it("waits for the picker and every chunk ACK before completing", async () => {
    const { messages, reply, bridge, previousHandler } =
      installDownloadBridge();
    const bytes = Uint8Array.from(
      { length: 48 * 1024 + 1 },
      (_, index) => index % 256,
    );
    const download = saveBlob(
      new Blob([bytes], { type: "text/markdown;charset=utf-8" }),
      "report.md",
    );
    const start = messages[0];
    expect(start).toMatchObject({
      type: "start",
      filename: "report.md",
      mimeType: "text/markdown",
      size: bytes.length,
    });
    expect(messages).toHaveLength(1);
    reply({ type: "ready", id: start.id });
    await waitFor(() => expect(messages).toHaveLength(2));
    expect(messages[1]).toMatchObject({ type: "chunk", index: 0 });
    expect(messages[1].data).toHaveLength(64 * 1024);
    reply({ type: "ack", id: start.id, index: 99 });
    await Promise.resolve();
    expect(messages).toHaveLength(2);
    reply({ type: "ack", id: start.id, index: 0 });
    await waitFor(() => expect(messages).toHaveLength(3));
    expect(messages[2]).toMatchObject({ type: "chunk", index: 1 });
    const restored = Uint8Array.from(
      atob(messages[1].data!) + atob(messages[2].data!),
      (character) => character.charCodeAt(0),
    );
    expect(restored).toEqual(bytes);
    reply({ type: "ack", id: start.id, index: 1 });
    await waitFor(() =>
      expect(messages[3]).toMatchObject({ type: "finish", id: start.id }),
    );
    reply({ type: "complete", id: start.id });
    await download;
    expect(bridge.onmessage).toBe(previousHandler);
  });

  it("supports empty files and forwards unrelated replies to the prior handler", async () => {
    const { messages, reply, previousHandler } = installDownloadBridge();
    const download = saveBlob(new Blob([]), "empty.txt");
    const id = messages[0].id;
    reply({ type: "ready", id: "another-download" });
    expect(previousHandler).toHaveBeenCalledTimes(1);
    expect(messages).toHaveLength(1);
    reply({ type: "ready", id });
    await waitFor(() =>
      expect(messages[1]).toMatchObject({ type: "finish", id }),
    );
    reply({ type: "complete", id });
    await download;
  });

  it("rejects native errors and restores the previous message handler", async () => {
    const { messages, reply, bridge, previousHandler } =
      installDownloadBridge();
    const download = saveBlob(new Blob(["x"]), "x.txt");
    reply({ type: "error", id: messages[0].id, message: "Storage full" });
    await expect(download).rejects.toThrow("Storage full");
    expect(bridge.onmessage).toBe(previousHandler);
    expect(messages).toHaveLength(1);
  });

  it("treats picker cancellation as cancellation without sending file content", async () => {
    const { messages, reply, bridge, previousHandler } =
      installDownloadBridge();
    const download = saveBlob(new Blob(["private content"]), "x.txt");
    reply({ type: "cancelled", id: messages[0].id });
    await expect(download).rejects.toMatchObject({ name: "AbortError" });
    expect(messages).toHaveLength(1);
    expect(bridge.onmessage).toBe(previousHandler);
  });

  it("aborts the native transfer on caller cancellation and permits another download", async () => {
    const { messages, reply, bridge, previousHandler } =
      installDownloadBridge();
    const controller = new AbortController();
    const download = saveBlob(new Blob(["x"]), "x.txt", {
      signal: controller.signal,
    });
    const id = messages[0].id;
    controller.abort();
    await expect(download).rejects.toMatchObject({ name: "AbortError" });
    expect(messages[1]).toEqual({ type: "cancel", id });
    expect(bridge.onmessage).toBe(previousHandler);
    const retry = saveBlob(new Blob([]), "retry.txt");
    const retryID = messages[2].id;
    reply({ type: "cancelled", id: retryID });
    await expect(retry).rejects.toMatchObject({ name: "AbortError" });
  });

  it("rejects a concurrent transfer without taking over its handler", async () => {
    const { messages, reply, bridge } = installDownloadBridge();
    const first = saveBlob(new Blob(["first"]), "first.txt");
    const handler = bridge.onmessage;
    await expect(saveBlob(new Blob(["second"]), "second.txt")).rejects.toThrow(
      "already in progress",
    );
    expect(bridge.onmessage).toBe(handler);
    expect(messages).toHaveLength(1);
    reply({ type: "cancelled", id: messages[0].id });
    await expect(first).rejects.toMatchObject({ name: "AbortError" });
  });

  it("times out a stalled picker and cancels its native request", async () => {
    vi.useFakeTimers();
    const { messages, bridge, previousHandler } = installDownloadBridge();
    const download = saveBlob(new Blob(["x"]), "x.txt");
    const outcome = expect(download).rejects.toThrow("timed out");
    await vi.advanceTimersByTimeAsync(120000);
    expect(messages).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(10000);
    await outcome;
    expect(messages[1]).toEqual({ type: "cancel", id: messages[0].id });
    expect(bridge.onmessage).toBe(previousHandler);
  });

  it("allows the provider copy to run for 120 seconds before timing out at 130 seconds", async () => {
    vi.useFakeTimers();
    const { messages, reply, bridge, previousHandler } =
      installDownloadBridge();
    const download = saveBlob(new Blob([]), "empty.txt");
    const outcome = expect(download).rejects.toThrow("timed out");
    const id = messages[0].id;
    reply({ type: "ready", id });
    await vi.advanceTimersByTimeAsync(0);
    expect(messages[1]).toEqual({ type: "finish", id });
    await vi.advanceTimersByTimeAsync(120000);
    expect(messages).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(10000);
    await outcome;
    expect(messages[2]).toEqual({ type: "cancel", id });
    expect(bridge.onmessage).toBe(previousHandler);
  });

  it("still times out an unacknowledged chunk after 30 seconds", async () => {
    vi.useFakeTimers();
    vi.spyOn(Blob.prototype, "arrayBuffer").mockResolvedValue(
      Uint8Array.of(120).buffer,
    );
    const { messages, reply, bridge, previousHandler } =
      installDownloadBridge();
    const download = saveBlob(new Blob(["x"]), "x.txt");
    const outcome = expect(download).rejects.toThrow("timed out");
    const id = messages[0].id;
    reply({ type: "ready", id });
    await vi.advanceTimersByTimeAsync(0);
    expect(messages[1]).toMatchObject({ type: "chunk", id, index: 0 });
    await vi.advanceTimersByTimeAsync(29999);
    expect(messages).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    await outcome;
    expect(messages[2]).toEqual({ type: "cancel", id });
    expect(bridge.onmessage).toBe(previousHandler);
  });

  it("rejects over-50-MiB files before requesting a save location", async () => {
    const { messages, bridge, previousHandler } = installDownloadBridge();
    const blob = new Blob(["x"]);
    Object.defineProperty(blob, "size", { value: 50 * 1024 * 1024 + 1 });
    await expect(saveBlob(blob, "large.bin")).rejects.toThrow("50 MiB");
    expect(messages).toHaveLength(0);
    expect(bridge.onmessage).toBe(previousHandler);
  });

  it("cancels when the web page leaves", async () => {
    const { messages } = installDownloadBridge();
    const download = saveBlob(new Blob(["x"]), "x.txt");
    window.dispatchEvent(new Event("pagehide"));
    await expect(download).rejects.toMatchObject({ name: "AbortError" });
    expect(messages[1]).toEqual({ type: "cancel", id: messages[0].id });
  });

  it("retains a native failure that arrives between acknowledged steps", async () => {
    const { messages, reply } = installDownloadBridge();
    const download = saveBlob(new Blob(["x"]), "x.txt");
    const id = messages[0].id;
    reply({ type: "ready", id });
    reply({ type: "error", id, message: "Save destination disappeared" });
    await expect(download).rejects.toThrow("Save destination disappeared");
    expect(messages).toHaveLength(1);
  });

  it("does not overwrite a new handler installed by another consumer", async () => {
    const { messages, bridge } = installDownloadBridge();
    const controller = new AbortController();
    const download = saveBlob(new Blob(["x"]), "x.txt", {
      signal: controller.signal,
    });
    const nextHandler = vi.fn();
    bridge.onmessage = nextHandler;
    controller.abort();
    await expect(download).rejects.toMatchObject({ name: "AbortError" });
    expect(messages[1]).toEqual({ type: "cancel", id: messages[0].id });
    expect(bridge.onmessage).toBe(nextHandler);
  });
});

import { getGetV1GetSpecificGraphResponseMock200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useGraphStore } from "../../stores/graphStore";
import { isSaveCancelled, waitForLoadedGraph } from "../waitForLoadedGraph";

const graph = getGetV1GetSpecificGraphResponseMock200({
  id: "graph-1",
  version: 2,
});

// What useFlow does as the builder loads a URL and then fills the canvas.
function loadCanvas(version: number | null, loaded = graph) {
  useGraphStore.getState().requestLoad("graph-1", version);
  useGraphStore.getState().setLoadedGraph(loaded);
}

afterEach(() => {
  vi.useRealTimers();
  useGraphStore.getState().reset();
});

describe("waitForLoadedGraph", () => {
  test("resolves at once when the canvas already shows the graph", async () => {
    vi.useFakeTimers();
    loadCanvas(2);

    const loaded = waitForLoadedGraph("graph-1", 2);
    await vi.advanceTimersByTimeAsync(30_000);

    await expect(loaded).resolves.toBe(graph);
  });

  test("waits for the canvas to show that version of the graph", async () => {
    loadCanvas(2, { ...graph, version: 1 });

    const loaded = waitForLoadedGraph("graph-1", 2);
    useGraphStore.getState().setLoadedGraph(graph);

    await expect(loaded).resolves.toBe(graph);
  });

  test("with no version asked for, waits for a load made for the latest version", async () => {
    loadCanvas(1, { ...graph, version: 1 });
    useGraphStore.getState().requestLoad("graph-1", null);

    const loaded = waitForLoadedGraph("graph-1", null);
    useGraphStore.getState().setLoadedGraph(graph);

    await expect(loaded).resolves.toBe(graph);
  });

  test("still counts a load as the latest once the URL is given the version it resolved to", async () => {
    useGraphStore.getState().requestLoad("graph-1", null);
    const loaded = waitForLoadedGraph("graph-1", null);

    useGraphStore.getState().resolveLatestVersion("graph-1", 2);
    loadCanvas(2);

    await expect(loaded).resolves.toBe(graph);
  });

  test("is cancelled when a version is picked while it waits for the latest", async () => {
    useGraphStore.getState().requestLoad("graph-1", null);
    const beforeResolving = waitForLoadedGraph("graph-1", null);
    useGraphStore.getState().requestLoad("graph-1", 2);

    useGraphStore.getState().requestLoad("graph-1", null);
    useGraphStore.getState().resolveLatestVersion("graph-1", 2);
    const afterResolving = waitForLoadedGraph("graph-1", null);
    useGraphStore.getState().requestLoad("graph-1", 1);

    expect(isSaveCancelled(await beforeResolving.catch((e) => e))).toBe(true);
    expect(isSaveCancelled(await afterResolving.catch((e) => e))).toBe(true);
  });

  test("fails at once when the canvas can't load the workflow", async () => {
    vi.useFakeTimers();
    const loaded = waitForLoadedGraph("graph-1", 2);
    useGraphStore.getState().setLoadFailed(true);

    await expect(loaded).rejects.toThrow(
      "Your workflow couldn't be loaded. Reload the page and try again.",
    );
  });

  test("is cancelled when the builder moves to another workflow or version first", async () => {
    useGraphStore.getState().requestLoad("graph-1", 1);
    const otherWorkflow = waitForLoadedGraph("graph-1", 1);
    useGraphStore.getState().requestLoad("graph-2", 1);

    useGraphStore.getState().requestLoad("graph-1", 1);
    const otherVersion = waitForLoadedGraph("graph-1", 1);
    useGraphStore.getState().requestLoad("graph-1", 2);

    expect(isSaveCancelled(await otherWorkflow.catch((e) => e))).toBe(true);
    expect(isSaveCancelled(await otherVersion.catch((e) => e))).toBe(true);
  });

  test("is cancelled when the builder closes, without touching the next builder's", async () => {
    useGraphStore.getState().requestLoad("graph-1", 2);
    const beforeClosing = waitForLoadedGraph("graph-1", 2);
    useGraphStore.getState().reset();

    useGraphStore.getState().requestLoad("graph-1", 2);
    const afterReopening = waitForLoadedGraph("graph-1", 2);
    useGraphStore.getState().setLoadedGraph(graph);

    expect(isSaveCancelled(await beforeClosing.catch((e) => e))).toBe(true);
    await expect(afterReopening).resolves.toBe(graph);
  });
});

import { getGetV1GetSpecificGraphResponseMock200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useGraphStore } from "../../stores/graphStore";
import { isSaveCancelled, waitForLoadedGraph } from "../waitForLoadedGraph";

const graph = getGetV1GetSpecificGraphResponseMock200({
  id: "graph-1",
  version: 2,
});

afterEach(() => {
  vi.useRealTimers();
  useGraphStore.getState().reset();
  useGraphStore.getState().setLoadSignal(new AbortController().signal);
});

describe("waitForLoadedGraph", () => {
  test("resolves at once when the canvas already shows the graph", async () => {
    vi.useFakeTimers();
    useGraphStore.getState().setLoadedGraph(graph);

    const loaded = waitForLoadedGraph("graph-1", 2);
    await vi.advanceTimersByTimeAsync(30_000);

    await expect(loaded).resolves.toBe(graph);
  });

  test("waits for the canvas to show that version of the graph", async () => {
    useGraphStore.getState().setLoadedGraph({ ...graph, version: 1 });

    const loaded = waitForLoadedGraph("graph-1", 2);
    useGraphStore.getState().setLoadedGraph(graph);

    await expect(loaded).resolves.toBe(graph);
  });

  test("fails at once when the canvas can't load the workflow's blocks", async () => {
    vi.useFakeTimers();
    const loaded = waitForLoadedGraph("graph-1", 2);
    useGraphStore.getState().setLoadFailed(true);

    await expect(loaded).rejects.toThrow(
      "Your workflow couldn't be loaded. Reload the page and try again.",
    );
  });

  test("is cancelled when the builder leaves the workflow before it loads", async () => {
    const controller = new AbortController();
    useGraphStore.getState().setLoadSignal(controller.signal);

    const loaded = waitForLoadedGraph("graph-1", 2);
    controller.abort();

    expect(isSaveCancelled(await loaded.catch((error) => error))).toBe(true);
  });
});

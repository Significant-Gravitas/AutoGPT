import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import * as helpers from "../../../helpers";
import { useNodeDataViewer } from "../useNodeDataViewer";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

vi.mock("../../../helpers", async (importOriginal) => {
  const actual = await importOriginal<typeof helpers>();
  return { ...actual, createOutputItems: vi.fn(actual.createOutputItems) };
});

function createResult(execId: string, output: unknown[]): NodeExecutionResult {
  return {
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    graph_exec_id: "graph-exec-1",
    node_exec_id: execId,
    node_id: "1",
    block_id: "block-1",
    status: "COMPLETED",
    input_data: {},
    output_data: { result: output },
    add_time: new Date("2026-01-05T09:00:00.000Z"),
    queue_time: new Date("2026-01-05T09:00:00.000Z"),
    start_time: new Date("2026-01-05T09:00:01.000Z"),
    end_time: new Date("2026-01-05T09:00:02.000Z"),
  };
}

interface Props {
  data: unknown;
  executionResults: NodeExecutionResult[];
}

function renderViewer(initialProps: Props) {
  return renderHook(
    ({ data, executionResults }: Props) =>
      useNodeDataViewer(data, "result", "exec-2", executionResults, "output"),
    { initialProps },
  );
}

beforeEach(() => {
  vi.mocked(helpers.createOutputItems).mockClear();
});

describe("useNodeDataViewer", () => {
  it("builds the items newest execution first", () => {
    const executionResults = [
      createResult("exec-1", ["first"]),
      createResult("exec-2", ["second", "third"]),
    ];
    const { result } = renderViewer({
      data: ["second", "third"],
      executionResults,
    });

    expect(result.current.outputItems.map((item) => item.value)).toEqual([
      "second",
      "third",
    ]);
    expect(result.current.outputItems[0].label).toBe("Result");
    expect(result.current.groupedExecutions.map((g) => g.execId)).toEqual([
      "exec-2",
      "exec-1",
    ]);
    expect(result.current.totalGroupedItems).toBe(3);
  });

  it("does not rebuild the items when its node re-renders with the same data", () => {
    const data = ["value"];
    const executionResults = [createResult("exec-1", ["value"])];
    const { result, rerender } = renderViewer({ data, executionResults });
    const callsAfterMount = vi.mocked(helpers.createOutputItems).mock.calls
      .length;
    const firstItems = result.current.outputItems;

    rerender({ data, executionResults });
    rerender({ data, executionResults });

    expect(vi.mocked(helpers.createOutputItems).mock.calls.length).toBe(
      callsAfterMount,
    );
    expect(result.current.outputItems).toBe(firstItems);
  });

  it("rebuilds the items when new output arrives", () => {
    const { result, rerender } = renderViewer({
      data: ["first"],
      executionResults: [createResult("exec-1", ["first"])],
    });

    rerender({
      data: ["first", "second"],
      executionResults: [
        createResult("exec-1", ["first"]),
        createResult("exec-2", ["second"]),
      ],
    });

    expect(result.current.outputItems.map((item) => item.value)).toEqual([
      "first",
      "second",
    ]);
    expect(result.current.groupedExecutions.map((g) => g.execId)).toEqual([
      "exec-2",
      "exec-1",
    ]);
  });
});

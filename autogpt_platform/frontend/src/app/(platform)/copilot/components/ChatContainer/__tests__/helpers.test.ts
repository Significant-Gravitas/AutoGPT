import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";
import { getLatestTurnOverwrites } from "../helpers";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

function write(fileId: string, replaced?: unknown, output?: unknown) {
  return {
    type: "tool-write_workspace_file",
    toolCallId: `call-${fileId}`,
    state: "output-available",
    input: {},
    output: output ?? {
      file_id: fileId,
      name: "report.md",
      mime_type: "text/markdown",
      size_bytes: 12,
      ...(replaced ? { replaced_file_id: replaced } : {}),
    },
  } as Message["parts"][number];
}

function message(role: Message["role"], ...parts: Message["parts"]): Message {
  return { id: crypto.randomUUID(), role, parts };
}

function replacedIds(messages: Message[]) {
  return getLatestTurnOverwrites(messages)
    .map((o) => [o.replacedFileId, o.next.id])
    .sort();
}

const TOO_BIG = 11 * 1024 * 1024;

function bigWrite(fileId: string, replaced: string) {
  return write(fileId, undefined, {
    file_id: fileId,
    replaced_file_id: replaced,
    name: "data.csv",
    mime_type: "text/csv",
    size_bytes: TOO_BIG,
  });
}

describe("getLatestTurnOverwrites", () => {
  it("builds the ref the panel opens from the write's output", () => {
    const [overwrite] = getLatestTurnOverwrites([
      message("assistant", write("v2", "v1")),
    ]);
    expect(overwrite).toEqual({
      replacedFileId: "v1",
      next: {
        id: "v2",
        title: "report.md",
        mimeType: "text/markdown",
        sourceUrl: "/api/proxy/api/workspace/files/v2/download",
        origin: "agent",
        sizeBytes: 12,
      },
    });
  });

  it("points every version of a file rewritten twice at the last one", () => {
    expect(
      replacedIds([message("assistant", write("v2", "v1"), write("v3", "v2"))]),
    ).toEqual([
      ["v1", "v3"],
      ["v2", "v3"],
    ]);
  });

  it("reads every message of the latest turn and none of an earlier one", () => {
    expect(
      replacedIds([
        message("assistant", write("old2", "old1")),
        message("user"),
        message("assistant", write("v2", "v1")),
        message("assistant", write("v3", "v2")),
      ]),
    ).toEqual([
      ["v1", "v3"],
      ["v2", "v3"],
    ]);
  });

  it("still reads the latest turn once the user has sent the next message", () => {
    expect(
      replacedIds([message("assistant", write("v2", "v1")), message("user")]),
    ).toEqual([["v1", "v2"]]);
  });

  it("reads an output that arrives as a JSON string", () => {
    const output = JSON.stringify({
      file_id: "v2",
      replaced_file_id: "v1",
      name: "report.md",
    });
    expect(
      replacedIds([message("assistant", write("v2", undefined, output))]),
    ).toEqual([["v1", "v2"]]);
  });

  it("ignores a new file, a failed write, and a write still running", () => {
    expect(
      replacedIds([
        message(
          "assistant",
          write("new"),
          write("err", undefined, { type: "error", message: "No" }),
          {
            type: "tool-write_workspace_file",
            toolCallId: "running",
            state: "input-available",
            input: {},
          } as Message["parts"][number],
          { type: "text", text: "Done" },
        ),
      ]),
    ).toEqual([]);
  });

  it("leaves out a new version too big for the panel to preview", () => {
    expect(replacedIds([message("assistant", bigWrite("v2", "v1"))])).toEqual(
      [],
    );
  });

  it("follows past a too-big version to a later one it can preview", () => {
    expect(
      replacedIds([
        message("assistant", bigWrite("v2", "v1"), write("v3", "v2")),
      ]),
    ).toEqual([
      ["v1", "v3"],
      ["v2", "v3"],
    ]);
  });

  it("follows nothing when a file's last version is too big", () => {
    expect(
      replacedIds([
        message("assistant", write("v2", "v1"), bigWrite("v3", "v2")),
      ]),
    ).toEqual([]);
  });

  it("ignores another tool's output that has the same fields", () => {
    expect(
      replacedIds([
        message("assistant", {
          type: "tool-other",
          toolCallId: "other",
          state: "output-available",
          input: {},
          output: { file_id: "v2", replaced_file_id: "v1" },
        } as Message["parts"][number]),
      ]),
    ).toEqual([]);
  });
});

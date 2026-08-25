import { describe, expect, it } from "vitest";
import {
  WORKSPACE_FILE_PATTERN,
  extractWorkspaceArtifacts,
  filePartToArtifactRef,
  getLatestCompactionPhase,
  getLatestCompactionStats,
  getMessageArtifacts,
  getMostRecentArtifact,
  parseSpecialMarkers,
  resolveWorkspaceUrls,
} from "../helpers";
import type { MessagePart } from "../helpers";
import type { FileUIPart, UIDataTypes, UIMessage, UITools } from "ai";

function textPart(text: string): MessagePart {
  return { type: "text", text } as MessagePart;
}

function stepStartPart(): MessagePart {
  return { type: "step-start" } as MessagePart;
}

describe("parseSpecialMarkers", () => {
  it("returns null marker for plain text", () => {
    const result = parseSpecialMarkers("Hello world");
    expect(result.markerType).toBeNull();
    expect(result.cleanText).toBe("Hello world");
  });

  it("detects error marker", () => {
    const result = parseSpecialMarkers(
      "Some preamble [__COPILOT_ERROR_f7a1__] Something went wrong",
    );
    expect(result.markerType).toBe("error");
    expect(result.markerText).toBe("Something went wrong");
  });

  it("detects retryable error marker", () => {
    const result = parseSpecialMarkers(
      "[__COPILOT_RETRYABLE_ERROR_a9c2__] Timeout reached",
    );
    expect(result.markerType).toBe("retryable_error");
    expect(result.markerText).toBe("Timeout reached");
  });

  it("detects system marker", () => {
    const result = parseSpecialMarkers(
      "[__COPILOT_SYSTEM_e3b0__] Session expired",
    );
    expect(result.markerType).toBe("system");
    expect(result.markerText).toBe("Session expired");
  });

  it("retryable takes precedence over regular error when both present", () => {
    const text =
      "[__COPILOT_RETRYABLE_ERROR_a9c2__] Retryable issue [__COPILOT_ERROR_f7a1__] Also error";
    const result = parseSpecialMarkers(text);
    expect(result.markerType).toBe("retryable_error");
  });

  it("strips marker from cleanText", () => {
    const result = parseSpecialMarkers(
      "Preamble text [__COPILOT_SYSTEM_e3b0__] System message",
    );
    expect(result.cleanText).toBe("Preamble text");
  });
});

describe("extractWorkspaceArtifacts", () => {
  it("extracts a single workspace:// link with its markdown title", () => {
    const text =
      "See [the report](workspace://550e8400-e29b-41d4-a716-446655440000) for details.";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("550e8400-e29b-41d4-a716-446655440000");
    expect(out[0].title).toBe("the report");
    expect(out[0].origin).toBe("agent");
  });

  it("falls back to a synthetic title when the URI isn't wrapped in link markdown", () => {
    const text = "raw workspace://abc12345-0000-0000-0000-000000000000 link";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
    expect(out[0].title).toBe("File abc12345");
  });

  it("skips URIs inside image markdown so images don't double-render", () => {
    const text =
      "![chart](workspace://abc12345-0000-0000-0000-000000000000#image/png)";
    expect(extractWorkspaceArtifacts(text)).toEqual([]);
  });

  it("still extracts non-image links when image links are also present", () => {
    const text =
      "![chart](workspace://aaaaaaaa-0000-0000-0000-000000000000#image/png) " +
      "and [doc](workspace://bbbbbbbb-0000-0000-0000-000000000000)";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("bbbbbbbb-0000-0000-0000-000000000000");
  });

  it("deduplicates repeated references to the same artifact id", () => {
    const text =
      "[A](workspace://11111111-0000-0000-0000-000000000000) and " +
      "[A again](workspace://11111111-0000-0000-0000-000000000000)";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
  });

  it("returns empty when no workspace URIs are present", () => {
    expect(extractWorkspaceArtifacts("plain text, no links")).toEqual([]);
  });

  it("picks up the mime hint from the URI fragment", () => {
    const text =
      "![v](workspace://cccccccc-0000-0000-0000-000000000000#video/mp4) " +
      "[d](workspace://dddddddd-0000-0000-0000-000000000000#application/pdf)";
    const out = extractWorkspaceArtifacts(text);
    expect(out).toHaveLength(1);
    expect(out[0].mimeType).toBe("application/pdf");
  });
});

describe("filePartToArtifactRef", () => {
  it("returns null without a url", () => {
    expect(
      filePartToArtifactRef({ type: "file", url: "", filename: "x" } as any),
    ).toBeNull();
  });

  it("returns null for URLs that don't match the workspace file pattern", () => {
    expect(
      filePartToArtifactRef({
        type: "file",
        url: "https://example.com/file.txt",
        filename: "file.txt",
      } as any),
    ).toBeNull();
  });

  it("extracts id from the workspace proxy URL", () => {
    const ref = filePartToArtifactRef({
      type: "file",
      url: "/api/proxy/api/workspace/files/550e8400-e29b-41d4-a716-446655440000/download",
      filename: "report.pdf",
      mediaType: "application/pdf",
    } as any);
    expect(ref?.id).toBe("550e8400-e29b-41d4-a716-446655440000");
    expect(ref?.title).toBe("report.pdf");
    expect(ref?.mimeType).toBe("application/pdf");
  });

  it("defaults origin to user-upload but accepts an override", () => {
    const url =
      "/api/proxy/api/workspace/files/550e8400-e29b-41d4-a716-446655440000/download";
    const defaulted = filePartToArtifactRef({
      type: "file",
      url,
      filename: "a.txt",
    } as any);
    expect(defaulted?.origin).toBe("user-upload");
    const overridden = filePartToArtifactRef(
      { type: "file", url, filename: "a.txt" } as any,
      "agent",
    );
    expect(overridden?.origin).toBe("agent");
  });
});

// ----- Custom fileUrlBuilder threading -----------------------------------
// The public-share viewer threads a token-aware URL builder through
// these helpers so anonymous readers can render file references that
// hit the public allowlist-gated download endpoint instead of the
// auth'd workspace one.  These tests pin the contract.

describe("extractWorkspaceArtifacts with custom fileUrlBuilder", () => {
  const FILE_ID = "550e8400-e29b-41d4-a716-446655440000";

  it("routes sourceUrl through the supplied builder", () => {
    const text = `See [report](workspace://${FILE_ID}) for details.`;
    const builder = (id: string) => `/share/files/${id}.dl`;
    const out = extractWorkspaceArtifacts(text, builder);
    expect(out).toHaveLength(1);
    expect(out[0].sourceUrl).toBe(`/share/files/${FILE_ID}.dl`);
  });

  it("default builder produces the workspace-file URL", () => {
    const text = `[report](workspace://${FILE_ID})`;
    const out = extractWorkspaceArtifacts(text);
    expect(out[0].sourceUrl).toContain(`/files/${FILE_ID}/download`);
  });
});

describe("resolveWorkspaceUrls with custom fileUrlBuilder", () => {
  const FILE_ID = "550e8400-e29b-41d4-a716-446655440000";

  it("rewrites image syntax using the supplied builder", () => {
    const text = `![pic](workspace://${FILE_ID}#image/png)`;
    const builder = (id: string) => `/share/files/${id}.png`;
    const out = resolveWorkspaceUrls(text, builder);
    expect(out).toBe(`![pic](/share/files/${FILE_ID}.png)`);
  });

  it("rewrites link syntax to absolute URL with origin prefix", () => {
    const text = `Open [the file](workspace://${FILE_ID}) here.`;
    const builder = (id: string) => `/share/files/${id}.dl`;
    const out = resolveWorkspaceUrls(text, builder);
    // jsdom's window.location.origin is "http://localhost:3000".
    expect(out).toContain(`(http://localhost:3000/share/files/${FILE_ID}.dl)`);
  });

  it("default builder rewrites workspace:// to the workspace endpoint", () => {
    const text = `![pic](workspace://${FILE_ID})`;
    const out = resolveWorkspaceUrls(text);
    expect(out).toMatch(/api\/workspace\/files\/.*\/download/);
  });

  it("video MIME hint produces video: alt prefix", () => {
    const text = `![demo](workspace://${FILE_ID}#video/mp4)`;
    const builder = (id: string) => `/share/files/${id}.mp4`;
    const out = resolveWorkspaceUrls(text, builder);
    expect(out).toBe(`![video:demo](/share/files/${FILE_ID}.mp4)`);
  });
});

describe("filePartToArtifactRef with custom pattern", () => {
  const FILE_ID = "550e8400-e29b-41d4-a716-446655440000";
  const file: FileUIPart = {
    type: "file",
    filename: "report.png",
    mediaType: "image/png",
    url: `/api/proxy/api/public/shared/chats/some-token/files/${FILE_ID}/download`,
  };

  it("default pattern (workspace-file) rejects public-share URL", () => {
    expect(filePartToArtifactRef(file)).toBeNull();
  });

  it("custom pattern matching the public-share URL extracts the file ID", () => {
    const pattern =
      /\/api\/proxy\/api\/public\/shared\/chats\/[^/]+\/files\/([a-f0-9-]+)\/download/;
    const ref = filePartToArtifactRef(file, "agent", pattern);
    expect(ref?.id).toBe(FILE_ID);
    expect(ref?.title).toBe("report.png");
    expect(ref?.mimeType).toBe("image/png");
  });

  it("returns null when url has no file", () => {
    expect(
      filePartToArtifactRef({ ...file, url: "" } as FileUIPart),
    ).toBeNull();
  });

  it("WORKSPACE_FILE_PATTERN matches a workspace-file URL", () => {
    const url = `/api/proxy/api/workspace/files/${FILE_ID}/download`;
    expect(url.match(WORKSPACE_FILE_PATTERN)?.[1]).toBe(FILE_ID);
  });
});

type Message = UIMessage<unknown, UIDataTypes, UITools>;

const FILE_A = "550e8400-e29b-41d4-a716-446655440000";
const FILE_B = "660e8400-e29b-41d4-a716-446655440111";

function message(role: Message["role"], parts: MessagePart[]): Message {
  return { id: `m-${role}`, role, parts } as unknown as Message;
}

function filePart(fileId: string, filename: string): MessagePart {
  return {
    type: "file",
    filename,
    mediaType: "image/png",
    url: `/api/proxy/api/workspace/files/${fileId}/download`,
  } as unknown as MessagePart;
}

describe("getMessageArtifacts", () => {
  it("collects file-part artifacts before text artifacts", () => {
    const msg = message("assistant", [
      filePart(FILE_A, "from-file.png"),
      textPart(`Here is [doc](workspace://${FILE_B})`),
    ]);
    const out = getMessageArtifacts(msg);
    expect(out.map((a) => a.id)).toEqual([FILE_A, FILE_B]);
    expect(out[0].title).toBe("from-file.png");
  });

  it("does not double-count a file referenced as both a file part and in text", () => {
    const msg = message("assistant", [
      filePart(FILE_A, "rich.png"),
      textPart(`[again](workspace://${FILE_A})`),
    ]);
    const out = getMessageArtifacts(msg);
    expect(out).toHaveLength(1);
    // File-part metadata wins over the text-derived entry.
    expect(out[0].title).toBe("rich.png");
  });

  it("marks user-uploaded files with the user-upload origin", () => {
    const msg = message("user", [filePart(FILE_A, "upload.png")]);
    expect(getMessageArtifacts(msg)[0].origin).toBe("user-upload");
  });
});

describe("getMostRecentArtifact", () => {
  it("returns null when there are no artifacts", () => {
    expect(
      getMostRecentArtifact([message("assistant", [textPart("hi")])]),
    ).toBeNull();
  });

  it("returns the last file-part artifact scanning from the end", () => {
    const messages = [
      message("assistant", [filePart(FILE_A, "old.png")]),
      message("assistant", [filePart(FILE_B, "new.png")]),
    ];
    expect(getMostRecentArtifact(messages)?.id).toBe(FILE_B);
  });

  it("finds the most recent text-derived artifact", () => {
    const messages = [
      message("assistant", [textPart(`[a](workspace://${FILE_A})`)]),
    ];
    expect(getMostRecentArtifact(messages)?.id).toBe(FILE_A);
  });

  it("filters by origin when requested", () => {
    const messages = [
      message("user", [filePart(FILE_A, "upload.png")]),
      message("assistant", [textPart(`[b](workspace://${FILE_B})`)]),
    ];
    // Only agent-origin artifacts are eligible; the latest such one wins.
    expect(getMostRecentArtifact(messages, { origin: "agent" })?.id).toBe(
      FILE_B,
    );
    expect(getMostRecentArtifact(messages, { origin: "user-upload" })?.id).toBe(
      FILE_A,
    );
  });
});

function dataPart(type: string, data?: unknown): MessagePart {
  return { type, data } as unknown as MessagePart;
}

function compactionRowPart(
  state: string,
  output?: unknown,
  id = "compaction-1",
): MessagePart {
  return {
    type: "tool-context_compaction",
    state,
    toolCallId: id,
    toolName: "context_compaction",
    input: {},
    output,
  } as unknown as MessagePart;
}

describe("getLatestCompactionPhase", () => {
  const openRow = compactionRowPart("input-available");
  const summarizing = dataPart("data-compaction", {
    phase: "summarizing",
    tokensBefore: 128_000,
  });

  it("reads the latest phase behind the open row", () => {
    expect(
      getLatestCompactionPhase([stepStartPart(), openRow, summarizing]),
    ).toBe("summarizing");
  });

  it("survives ANY transient data part landing mid-compaction", () => {
    // data-pending-drained and data-mode-changed are real parts the backend
    // emits mid-turn; an enumerated deny-list dropped the phase (and the
    // bar) the moment one arrived.
    const parts = [
      openRow,
      summarizing,
      dataPart("data-pending-drained", { count: 1 }),
      dataPart("data-mode-changed", { mode: "chat" }),
      dataPart("data-status", { message: "working" }),
      dataPart("data-some-future-part"),
    ];
    expect(getLatestCompactionPhase(parts)).toBe("summarizing");
  });

  it("nulls the phase once real content lands past it", () => {
    expect(
      getLatestCompactionPhase([openRow, summarizing, textPart("Back to it.")]),
    ).toBeNull();
  });

  it("skips whitespace-only streaming text", () => {
    expect(
      getLatestCompactionPhase([openRow, summarizing, textPart("  ")]),
    ).toBe("summarizing");
  });

  it("drops the phase when the row was retired by the abort sentinel", () => {
    const abortedRow = compactionRowPart("output-available", "");
    expect(getLatestCompactionPhase([abortedRow, summarizing])).toBeNull();
  });

  it("drops the phase when the row closed with an error", () => {
    const failedRow = compactionRowPart("output-error");
    expect(getLatestCompactionPhase([failedRow, summarizing])).toBeNull();
  });

  it("ignores an earlier retired row when a later cycle is live", () => {
    const abortedRow = compactionRowPart(
      "output-available",
      "",
      "compaction-1",
    );
    const secondRow = compactionRowPart(
      "input-available",
      undefined,
      "compaction-2",
    );
    expect(
      getLatestCompactionPhase([
        abortedRow,
        textPart("hi"),
        secondRow,
        summarizing,
      ]),
    ).toBe("summarizing");
  });

  it("returns null with no compaction parts at all", () => {
    expect(getLatestCompactionPhase([textPart("hello")])).toBeNull();
  });
});

describe("getLatestCompactionStats", () => {
  it("merges stats across data-compaction parts, later phases winning", () => {
    const parts = [
      dataPart("data-compaction", {
        phase: "summarizing",
        tokensBefore: 128_000,
      }),
      dataPart("data-compaction", {
        phase: "rebuilding",
        tokensBefore: 128_000,
        tokensAfter: 31_000,
        messagesBefore: 412,
        messagesAfter: 38,
      }),
    ];
    expect(getLatestCompactionStats(parts)).toEqual({
      tokensBefore: 128_000,
      tokensAfter: 31_000,
      messagesBefore: 412,
      messagesAfter: 38,
    });
  });

  it("survives phase-only and junk payloads", () => {
    const parts = [
      dataPart("data-compaction", { phase: "summarizing" }),
      dataPart("data-compaction", null),
      dataPart("data-compaction"),
      textPart("hi"),
    ];
    expect(getLatestCompactionStats(parts)).toEqual({});
  });
});

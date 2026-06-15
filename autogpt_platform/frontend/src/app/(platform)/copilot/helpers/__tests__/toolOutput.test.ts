import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@sentry/nextjs", () => ({ captureMessage: vi.fn() }));

import * as Sentry from "@sentry/nextjs";
import {
  isCorruptedCardToolPart,
  isUnparseableJsonOutput,
  reportCorruptedToolOutput,
} from "../toolOutput";

beforeEach(() => {
  vi.clearAllMocks();
});

describe("isUnparseableJsonOutput", () => {
  it("returns true for truncated JSON", () => {
    expect(
      isUnparseableJsonOutput('{"type":"setup_requirements","message":"Conn'),
    ).toBe(true);
  });

  it("returns false for valid JSON", () => {
    expect(isUnparseableJsonOutput('{"type":"setup_requirements"}')).toBe(
      false,
    );
  });

  it("returns false for plain-text outputs", () => {
    expect(isUnparseableJsonOutput("Tool execution error: timeout")).toBe(
      false,
    );
  });

  it("returns false for empty and non-string outputs", () => {
    expect(isUnparseableJsonOutput("")).toBe(false);
    expect(isUnparseableJsonOutput(undefined)).toBe(false);
    expect(isUnparseableJsonOutput({ type: "setup_requirements" })).toBe(false);
  });
});

describe("isCorruptedCardToolPart", () => {
  const truncated = '{"type":"setup_requirements","message":"Conn';

  it("flags a card-capable tool with truncated JSON output", () => {
    expect(
      isCorruptedCardToolPart({
        type: "tool-run_block",
        state: "output-available",
        output: truncated,
      }),
    ).toBe(true);
  });

  it("ignores tools that never render cards", () => {
    expect(
      isCorruptedCardToolPart({
        type: "tool-search_docs",
        state: "output-available",
        output: truncated,
      }),
    ).toBe(false);
  });

  it("ignores non-completed and valid-output parts", () => {
    expect(
      isCorruptedCardToolPart({
        type: "tool-run_block",
        state: "output-error",
        output: truncated,
      }),
    ).toBe(false);
    expect(
      isCorruptedCardToolPart({
        type: "tool-run_block",
        state: "output-available",
        output: '{"type":"block_output","block_id":"b1"}',
      }),
    ).toBe(false);
  });
});

describe("reportCorruptedToolOutput", () => {
  it("reports once per toolCallId", () => {
    reportCorruptedToolOutput("call-dedupe-test", "tool-run_block");
    reportCorruptedToolOutput("call-dedupe-test", "tool-run_block");
    expect(vi.mocked(Sentry.captureMessage)).toHaveBeenCalledTimes(1);
  });
});

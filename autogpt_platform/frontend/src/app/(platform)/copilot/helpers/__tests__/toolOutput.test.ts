import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@sentry/nextjs", () => ({ captureMessage: vi.fn() }));

import * as Sentry from "@sentry/nextjs";
import {
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

describe("reportCorruptedToolOutput", () => {
  it("reports once per toolCallId", () => {
    reportCorruptedToolOutput("call-dedupe-test", "tool-run_block");
    reportCorruptedToolOutput("call-dedupe-test", "tool-run_block");
    expect(vi.mocked(Sentry.captureMessage)).toHaveBeenCalledTimes(1);
  });
});

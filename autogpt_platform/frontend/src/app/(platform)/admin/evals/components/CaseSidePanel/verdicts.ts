import type { Tone } from "@/components/styles/tones";

type Status = "pass" | "partial" | "fail" | "skipped";

export const VERDICT_WORDS: Record<Status, string> = {
  pass: "Pass",
  partial: "Partial",
  fail: "Fail",
  skipped: "Skipped",
};

export const VERDICT_TONES: Record<Status, Tone> = {
  pass: "success",
  partial: "warning",
  fail: "error",
  skipped: "quiet",
};

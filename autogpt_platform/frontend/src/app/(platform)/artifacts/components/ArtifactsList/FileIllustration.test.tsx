import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";

import {
  deriveBadgeLabel,
  FileIllustration,
  pickFileTypeKey,
} from "./FileIllustration";

describe("pickFileTypeKey", () => {
  test.each([
    ["image/png", "img"],
    ["video/mp4", "video"],
    ["application/pdf", "pdf"],
    ["text/html", "html"],
    ["application/xhtml+xml", "html"],
    [
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      "xls",
    ],
    ["application/vnd.ms-excel", "xls"],
    ["text/csv", "xls"],
    ["application/json", "json"],
    ["text/plain", "generic"],
    [undefined, "generic"],
  ])("pickFileTypeKey(%s) === %s", (mime, expected) => {
    expect(pickFileTypeKey(mime)).toBe(expected);
  });

  test("detects React apps by .jsx/.tsx extension", () => {
    expect(pickFileTypeKey("text/plain", "App.jsx")).toBe("react");
    expect(pickFileTypeKey(undefined, "Component.tsx")).toBe("react");
  });

  test("detects code by extension even when the MIME is misleading", () => {
    // `.ts` resolves to video/mp2t in the browser/OS MIME database.
    expect(pickFileTypeKey("video/mp2t", "main.ts")).toBe("code");
    expect(pickFileTypeKey("text/plain", "script.py")).toBe("code");
    expect(pickFileTypeKey(undefined, "query.sql")).toBe("code");
  });
});

describe("deriveBadgeLabel", () => {
  test("returns the uppercased file extension when present", () => {
    expect(deriveBadgeLabel("report.PDF", "application/pdf")).toBe("PDF");
    expect(deriveBadgeLabel("notes.md", "text/markdown")).toBe("MD");
  });

  test("truncates long extensions to four chars", () => {
    expect(deriveBadgeLabel("data.jsonld", "application/json")).toBe("JSON");
  });

  test("falls back to the mime subtype when no extension is present", () => {
    expect(deriveBadgeLabel("noext", "application/pdf")).toBe("PDF");
  });

  test("returns the FILE sentinel when neither name nor mime is informative", () => {
    expect(deriveBadgeLabel(undefined, undefined)).toBe("FILE");
    expect(deriveBadgeLabel("", "")).toBe("FILE");
  });
});

describe("FileIllustration", () => {
  test("renders the badge label override when provided", () => {
    render(<FileIllustration typeKey="pdf" label="CUSTOM" />);
    expect(screen.getByText("CUSTOM")).toBeDefined();
  });

  test("falls back to the type-config default label when no override is given", () => {
    render(<FileIllustration typeKey="pdf" />);
    // PDF config label
    expect(screen.getByText(/PDF/)).toBeDefined();
  });

  test("renders for every known type key", () => {
    for (const key of [
      "pdf",
      "xls",
      "json",
      "img",
      "html",
      "video",
      "react",
      "code",
      "generic",
    ] as const) {
      const { unmount } = render(<FileIllustration typeKey={key} />);
      unmount();
    }
  });
});

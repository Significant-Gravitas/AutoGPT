import { describe, expect, it } from "vitest";
import { sanitizeFilename, getUniqueFilename } from "../download-outputs";

describe("sanitizeFilename", () => {
  it("strips forward slashes", () => {
    expect(sanitizeFilename("path/to/file.txt")).toBe("path_to_file.txt");
  });

  it("strips backslashes", () => {
    expect(sanitizeFilename("path\\to\\file.txt")).toBe("path_to_file.txt");
  });

  it("replaces parent directory traversal", () => {
    const result = sanitizeFilename("../../etc/passwd");
    expect(result).not.toContain("/");
    expect(result).not.toContain("\\");
    expect(result).not.toContain("..");
    expect(result).not.toMatch(/^\./);
  });

  it("strips leading dots", () => {
    expect(sanitizeFilename(".gitignore")).toBe("gitignore");
    expect(sanitizeFilename("..hidden")).toBe("hidden");
    expect(sanitizeFilename("...triple")).toBe("triple");
  });

  it("returns 'file' for empty results", () => {
    expect(sanitizeFilename("")).toBe("file");
    expect(sanitizeFilename("...")).toBe("file");
    expect(sanitizeFilename(".")).toBe("file");
  });

  it("leaves safe filenames unchanged", () => {
    expect(sanitizeFilename("report.pdf")).toBe("report.pdf");
    expect(sanitizeFilename("image_001.png")).toBe("image_001.png");
  });
});

describe("getUniqueFilename", () => {
  it("returns the filename when not already used", () => {
    const used = new Set<string>();
    expect(getUniqueFilename("file.txt", used)).toBe("file.txt");
    expect(used.has("file.txt")).toBe(true);
  });

  it("appends a counter on collision", () => {
    const used = new Set<string>(["file.txt"]);
    expect(getUniqueFilename("file.txt", used)).toBe("file_1.txt");
    expect(used.has("file_1.txt")).toBe(true);
  });

  it("increments counter until unique", () => {
    const used = new Set<string>(["file.txt", "file_1.txt", "file_2.txt"]);
    expect(getUniqueFilename("file.txt", used)).toBe("file_3.txt");
  });

  it("handles filenames without extensions", () => {
    const used = new Set<string>(["README"]);
    expect(getUniqueFilename("README", used)).toBe("README_1");
  });

  it("sanitizes the filename before deduplication", () => {
    const used = new Set<string>();
    expect(getUniqueFilename("../evil.txt", used)).toBe("_evil.txt");
  });

  it("handles dotfiles by stripping leading dots first", () => {
    const used = new Set<string>();
    expect(getUniqueFilename(".gitignore", used)).toBe("gitignore");
  });
});

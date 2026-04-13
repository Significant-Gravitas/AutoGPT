import { describe, expect, it } from "vitest";
import { classifyArtifact } from "./helpers";

describe("classifyArtifact", () => {
  it("routes PDF by extension", () => {
    const c = classifyArtifact(null, "report.pdf");
    expect(c.type).toBe("pdf");
    expect(c.openable).toBe(true);
  });

  it("routes PDF by MIME when no extension matches", () => {
    const c = classifyArtifact("application/pdf", "noextension");
    expect(c.type).toBe("pdf");
  });

  it("routes JSX/TSX as react", () => {
    expect(classifyArtifact(null, "App.tsx").type).toBe("react");
    expect(classifyArtifact(null, "Comp.jsx").type).toBe("react");
  });

  it("routes code extensions to code", () => {
    expect(classifyArtifact(null, "script.py").type).toBe("code");
    expect(classifyArtifact(null, "main.go").type).toBe("code");
    expect(classifyArtifact(null, "Dockerfile.yml").type).toBe("code");
  });

  it("treats images as image (inline rendered)", () => {
    expect(classifyArtifact(null, "photo.png").type).toBe("image");
    expect(classifyArtifact("image/svg+xml", "unknown").type).toBe("image");
  });

  it("treats CSVs as csv with source toggle", () => {
    const c = classifyArtifact(null, "data.csv");
    expect(c.type).toBe("csv");
    expect(c.hasSourceToggle).toBe(true);
  });

  it("treats HTML as html with source toggle", () => {
    expect(classifyArtifact(null, "page.html").type).toBe("html");
    expect(classifyArtifact("text/html", "noext").type).toBe("html");
  });

  it("treats markdown as markdown", () => {
    expect(classifyArtifact(null, "README.md").type).toBe("markdown");
    expect(classifyArtifact("text/markdown", "x").type).toBe("markdown");
  });

  it("gates files > 10MB to download-only", () => {
    const c = classifyArtifact("text/plain", "big.txt", 20 * 1024 * 1024);
    expect(c.openable).toBe(false);
    expect(c.type).toBe("download-only");
  });

  it("treats binary/octet-stream MIME as download-only", () => {
    expect(classifyArtifact("application/zip", "a.zip").openable).toBe(false);
    expect(classifyArtifact("application/octet-stream", "x").openable).toBe(
      false,
    );
    expect(classifyArtifact("audio/mpeg", "track.mp3").openable).toBe(false);
  });

  it("defaults unknown extension+MIME to download-only (not text)", () => {
    // Regression: previously dumped binary as <pre>; now refuses to open.
    const c = classifyArtifact(null, "data.bin");
    expect(c.openable).toBe(false);
    expect(c.type).toBe("download-only");
  });

  it("is case-insensitive on extension", () => {
    expect(classifyArtifact(null, "image.PNG").type).toBe("image");
    expect(classifyArtifact(null, "Notes.MD").type).toBe("markdown");
  });

  it("prioritizes extension over MIME", () => {
    // Extension says CSV, MIME says plain text → extension wins.
    const c = classifyArtifact("text/plain", "data.csv");
    expect(c.type).toBe("csv");
  });

  it("classifies video/mp4 as video (previewable)", () => {
    const c = classifyArtifact("video/mp4", "clip.mp4");
    expect(c.type).toBe("video");
    expect(c.openable).toBe(true);
  });

  it("classifies video/webm as video (previewable)", () => {
    const c = classifyArtifact("video/webm", "clip.webm");
    expect(c.type).toBe("video");
    expect(c.openable).toBe(true);
  });

  // ── Extension coverage ────────────────────────────────────────────

  it("routes .htm as html (not just .html)", () => {
    const c = classifyArtifact(null, "page.htm");
    expect(c.type).toBe("html");
    expect(c.hasSourceToggle).toBe(true);
  });

  it("routes .json as json with source toggle", () => {
    const c = classifyArtifact(null, "config.json");
    expect(c.type).toBe("json");
    expect(c.hasSourceToggle).toBe(true);
  });

  it("routes .txt as text", () => {
    expect(classifyArtifact(null, "notes.txt").type).toBe("text");
  });

  it("routes .log as text", () => {
    expect(classifyArtifact(null, "server.log").type).toBe("text");
  });

  it("routes .mdx as markdown", () => {
    expect(classifyArtifact(null, "docs.mdx").type).toBe("markdown");
  });

  it("routes browser-safe video extensions to video", () => {
    for (const ext of [".mp4", ".webm", ".m4v"]) {
      const c = classifyArtifact(null, `clip${ext}`);
      expect(c.type).toBe("video");
      expect(c.openable).toBe(true);
    }
  });

  it("keeps legacy or unsupported video extensions download-only", () => {
    for (const ext of [".ogg", ".mov", ".avi", ".mkv", ".flv", ".mpeg"]) {
      const c = classifyArtifact(null, `clip${ext}`);
      expect(c.type).toBe("download-only");
      expect(c.openable).toBe(false);
    }
  });

  it("routes all code extensions to code", () => {
    const codeExts = [
      "main.js",
      "app.ts",
      "theme.scss",
      "legacy.less",
      "schema.graphql",
      "query.gql",
      "api.proto",
      "main.dart",
      "lib.rb",
      "server.rs",
      "App.java",
      "main.c",
      "util.cpp",
      "header.h",
      "Program.cs",
      "index.php",
      "main.swift",
      "App.kt",
      "run.sh",
      "start.bash",
      "prompt.zsh",
      "config.toml",
      "settings.ini",
      "app.cfg",
      "query.sql",
      "analysis.r",
      "game.lua",
      "script.pl",
      "Calc.scala",
    ];
    for (const file of codeExts) {
      expect(classifyArtifact(null, file).type).toBe("code");
    }
  });

  it("routes config filenames and extensions to code", () => {
    const configFiles = [
      ".env",
      ".env.local",
      "app.properties",
      "service.conf",
      ".gitignore",
      "Dockerfile",
      "Makefile",
    ];

    for (const file of configFiles) {
      expect(classifyArtifact(null, file).type).toBe("code");
    }
  });

  it("routes .jsonl as code for now", () => {
    const c = classifyArtifact(null, "events.jsonl");
    expect(c.type).toBe("code");
  });

  it("routes .tsv as csv/spreadsheet", () => {
    const c = classifyArtifact(null, "table.tsv");
    expect(c.type).toBe("csv");
    expect(c.hasSourceToggle).toBe(true);
  });

  it("routes .ics and .vcf as text", () => {
    expect(classifyArtifact(null, "calendar.ics").type).toBe("text");
    expect(classifyArtifact(null, "contact.vcf").type).toBe("text");
  });

  it("routes all image extensions to image", () => {
    for (const ext of [".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"]) {
      expect(classifyArtifact(null, `file${ext}`).type).toBe("image");
    }
  });

  // ── MIME fallback coverage ────────────────────────────────────────

  it("routes application/json MIME to json", () => {
    const c = classifyArtifact("application/json", "noext");
    expect(c.type).toBe("json");
  });

  it("routes text/x-* MIME prefix to code", () => {
    expect(classifyArtifact("text/x-python", "noext").type).toBe("code");
    expect(classifyArtifact("text/x-c", "noext").type).toBe("code");
    expect(classifyArtifact("text/x-java-source", "noext").type).toBe("code");
  });

  it("routes react MIME types to react", () => {
    expect(classifyArtifact("text/jsx", "noext").type).toBe("react");
    expect(classifyArtifact("text/tsx", "noext").type).toBe("react");
    expect(classifyArtifact("application/jsx", "noext").type).toBe("react");
    expect(classifyArtifact("application/x-typescript-jsx", "noext").type).toBe(
      "react",
    );
  });

  it("routes JavaScript/TypeScript MIME to code", () => {
    expect(classifyArtifact("application/javascript", "noext").type).toBe(
      "code",
    );
    expect(classifyArtifact("text/javascript", "noext").type).toBe("code");
    expect(classifyArtifact("application/typescript", "noext").type).toBe(
      "code",
    );
    expect(classifyArtifact("text/typescript", "noext").type).toBe("code");
  });

  it("routes XML MIME to code", () => {
    expect(classifyArtifact("application/xml", "noext").type).toBe("code");
    expect(classifyArtifact("text/xml", "noext").type).toBe("code");
  });

  it("routes text/x-markdown MIME to markdown", () => {
    expect(classifyArtifact("text/x-markdown", "noext").type).toBe("markdown");
  });

  it("routes text/csv MIME to csv", () => {
    expect(classifyArtifact("text/csv", "noext").type).toBe("csv");
  });

  it("routes TSV MIME to csv", () => {
    expect(classifyArtifact("text/tab-separated-values", "noext").type).toBe(
      "csv",
    );
  });

  it("routes unknown text/* MIME to text (not download-only)", () => {
    expect(classifyArtifact("text/rtf", "noext").type).toBe("text");
  });

  it("routes browser-safe image MIME types to image", () => {
    expect(classifyArtifact("image/avif", "noext").type).toBe("image");
  });

  it("keeps unsupported image MIME types download-only", () => {
    for (const mime of [
      "image/tiff",
      "image/x-portable-pixmap",
      "image/x-portable-graymap",
    ]) {
      const c = classifyArtifact(mime, "noext");
      expect(c.type).toBe("download-only");
      expect(c.openable).toBe(false);
    }
  });

  it("routes browser-safe video MIME types to video", () => {
    expect(classifyArtifact("video/mp4", "noext").type).toBe("video");
    expect(classifyArtifact("video/webm", "noext").type).toBe("video");
  });

  it("keeps legacy or unsupported video MIME types download-only", () => {
    for (const mime of [
      "video/x-msvideo",
      "video/x-flv",
      "video/mpeg",
      "video/quicktime",
      "video/x-matroska",
      "video/ogg",
    ]) {
      const c = classifyArtifact(mime, "noext");
      expect(c.type).toBe("download-only");
      expect(c.openable).toBe(false);
    }
  });

  // ── BINARY_MIMES coverage ────────────────────────────────────────

  it("treats all BINARY_MIMES entries as download-only", () => {
    const binaryMimes = [
      "application/zip",
      "application/x-zip-compressed",
      "application/gzip",
      "application/x-tar",
      "application/x-rar-compressed",
      "application/x-7z-compressed",
      "application/octet-stream",
      "application/x-executable",
      "application/x-msdos-program",
      "application/vnd.microsoft.portable-executable",
    ];
    for (const mime of binaryMimes) {
      const c = classifyArtifact(mime, "noext");
      expect(c.openable).toBe(false);
      expect(c.type).toBe("download-only");
    }
  });

  it("treats audio/* MIME as download-only", () => {
    expect(classifyArtifact("audio/mpeg", "noext").openable).toBe(false);
    expect(classifyArtifact("audio/wav", "noext").openable).toBe(false);
    expect(classifyArtifact("audio/ogg", "noext").openable).toBe(false);
  });

  // ── Size gate edge cases ──────────────────────────────────────────

  it("does NOT gate files at exactly 10MB (boundary is >10MB)", () => {
    const tenMB = 10 * 1024 * 1024;
    const c = classifyArtifact("text/plain", "exact.txt", tenMB);
    expect(c.type).toBe("text");
    expect(c.openable).toBe(true);
  });

  it("gates files at 10MB + 1 byte", () => {
    const overTenMB = 10 * 1024 * 1024 + 1;
    const c = classifyArtifact("text/plain", "big.txt", overTenMB);
    expect(c.type).toBe("download-only");
    expect(c.openable).toBe(false);
  });

  it("does not gate when sizeBytes is 0", () => {
    const c = classifyArtifact("text/plain", "empty.txt", 0);
    expect(c.type).toBe("text");
    expect(c.openable).toBe(true);
  });

  it("does not gate when sizeBytes is undefined", () => {
    const c = classifyArtifact("text/plain", "file.txt", undefined);
    expect(c.type).toBe("text");
    expect(c.openable).toBe(true);
  });

  // ── Extension over MIME priority ──────────────────────────────────

  it("extension wins over MIME for JSON (MIME says text, ext says json)", () => {
    const c = classifyArtifact("text/plain", "data.json");
    expect(c.type).toBe("json");
  });

  it("extension wins over MIME for markdown", () => {
    const c = classifyArtifact("text/plain", "README.md");
    expect(c.type).toBe("markdown");
  });

  // ── Null/missing inputs ───────────────────────────────────────────

  it("handles null MIME with no filename as download-only", () => {
    const c = classifyArtifact(null, undefined);
    expect(c.type).toBe("download-only");
  });

  it("handles null MIME with empty filename as download-only", () => {
    const c = classifyArtifact(null, "");
    expect(c.type).toBe("download-only");
  });

  it("handles known config files with no extension", () => {
    const c = classifyArtifact(null, "Makefile");
    expect(c.type).toBe("code");
  });

  // ── Exotic/compound extensions must NOT open the side panel ───────
  // These are real file types agents might produce. Every single one
  // must be download-only so we never try to render binary garbage.

  it("does not open .tar.gz (compound extension takes last segment)", () => {
    // getExtension("archive.tar.gz") → ".gz" which is not in EXT_KIND
    const c = classifyArtifact(null, "archive.tar.gz");
    expect(c.openable).toBe(false);
    expect(c.type).toBe("download-only");
  });

  it("does not open .tar.bz2", () => {
    const c = classifyArtifact(null, "archive.tar.bz2");
    expect(c.openable).toBe(false);
  });

  it("does not open .tar.xz", () => {
    const c = classifyArtifact(null, "archive.tar.xz");
    expect(c.openable).toBe(false);
  });

  it("does not open common binary formats", () => {
    const binaries = [
      "setup.exe",
      "library.dll",
      "image.iso",
      "installer.dmg",
      "package.deb",
      "package.rpm",
      "module.wasm",
      "Main.class",
      "module.pyc",
      "app.apk",
      "game.pak",
      "model.onnx",
      "weights.pt",
      "data.parquet",
      "archive.rar",
      "archive.7z",
      "disk.vhd",
      "disk.vmdk",
      "firmware.bin",
      "core.dump",
      "database.sqlite",
      "database.db",
      "index.idx",
    ];
    for (const file of binaries) {
      const c = classifyArtifact(null, file);
      expect(c.openable).toBe(false);
    }
  });

  it("does not open binary MIME types even with a misleading extension", () => {
    // Extension is unknown, MIME is binary
    const c = classifyArtifact("application/x-executable", "run.elf");
    expect(c.openable).toBe(false);
  });

  it("does not open files with random/made-up extensions", () => {
    const weirdExts = [
      "output.xyz",
      "data.foo",
      "file.asdf",
      "thing.blargh",
      "result.out",
      "x.1234",
    ];
    for (const file of weirdExts) {
      const c = classifyArtifact(null, file);
      expect(c.openable).toBe(false);
      expect(c.type).toBe("download-only");
    }
  });

  it("does not open font files", () => {
    for (const file of ["sans.ttf", "serif.otf", "icon.woff", "icon.woff2"]) {
      expect(classifyArtifact(null, file).openable).toBe(false);
    }
  });

  it("does not open certificate/key files", () => {
    // .pem and .key have no extension mapping and null MIME → download-only
    for (const file of ["cert.pem", "server.key", "ca.crt", "id.p12"]) {
      expect(classifyArtifact(null, file).openable).toBe(false);
    }
  });
});

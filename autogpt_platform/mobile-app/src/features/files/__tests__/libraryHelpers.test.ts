import type { WorkspaceFolder } from "../../../lib/api/workspace";
import {
  childFolders,
  folderPath,
  folderSummary,
  formatJson,
  libraryFilter,
  parseDelimited,
  tableFromText,
} from "../libraryHelpers";

function folder(id: string, name: string, parent: string | null, files = 0): WorkspaceFolder {
  return { id, name, parent_id: parent, file_count: files, updated_at: "2026-11-01T00:00:00Z" };
}

const FOLDERS = [
  folder("docs", "Documents", null, 4),
  folder("blog", "Blog", null, 5),
  folder("contracts", "Contracts", "docs", 3),
];

describe("libraryFilter", () => {
  it("maps the chips to the list's query", () => {
    expect(libraryFilter({ kind: "all" }, null, "")).toEqual({ q: null, origin: null, expertID: null, folderID: null });
    expect(libraryFilter({ kind: "you" }, "docs", " brief ")).toEqual({
      q: "brief",
      origin: "uploaded",
      expertID: null,
      folderID: "docs",
    });
    expect(libraryFilter({ kind: "expert", id: "e1" }, null, "")).toMatchObject({ expertID: "e1", origin: null });
  });
});

describe("folders", () => {
  it("lists a folder's children by name", () => {
    expect(childFolders(FOLDERS, null).map((item) => item.name)).toEqual(["Blog", "Documents"]);
    expect(childFolders(FOLDERS, "docs").map((item) => item.name)).toEqual(["Contracts"]);
  });

  it("walks from the top down to the folder", () => {
    expect(folderPath(FOLDERS, "contracts").map((item) => item.name)).toEqual(["Documents", "Contracts"]);
    expect(folderPath(FOLDERS, "gone")).toEqual([]);
  });

  it("stops on a loop in bad data", () => {
    const loop = [folder("a", "A", "b"), folder("b", "B", "a")];
    expect(folderPath(loop, "a").length).toBeLessThanOrEqual(3);
  });

  it("counts files and sub-folders", () => {
    expect(folderSummary(FOLDERS[0], FOLDERS)).toBe("4 files · 1 folder");
    expect(folderSummary(FOLDERS[2], FOLDERS)).toBe("3 files");
  });
});

describe("parseDelimited", () => {
  it("reads quoted cells with commas, quotes and line breaks", () => {
    expect(parseDelimited('name,note\r\n"Acme, Inc.","says ""hi""\nthere"\n', ",")).toEqual([
      ["name", "note"],
      ["Acme, Inc.", 'says "hi"\nthere'],
    ]);
  });

  it("reads tabs", () => {
    expect(parseDelimited("a\tb\n1\t2", "\t")).toEqual([
      ["a", "b"],
      ["1", "2"],
    ]);
  });
});

describe("tableFromText", () => {
  it("takes the first row as headings and spots number columns", () => {
    const table = tableFromText("Account,Value\nAcme,\"48,000\"\nGlobex,125500\n\n", { name: "pipeline.csv" });
    expect(table.header).toEqual(["Account", "Value"]);
    expect(table.rows).toHaveLength(2);
    expect(table.totalRows).toBe(2);
    expect(table.numeric).toEqual([false, true]);
  });

  it("keeps the first 500 rows but counts them all", () => {
    const text = ["n", ...Array.from({ length: 600 }, (_, index) => String(index))].join("\n");
    const table = tableFromText(text, { name: "big.csv" });
    expect(table.rows).toHaveLength(500);
    expect(table.totalRows).toBe(600);
  });

  it("splits on tabs for a TSV named without its extension", () => {
    const table = tableFromText("name\tvalue\nAcme\t12", {
      name: "report",
      mime_type: "text/tab-separated-values",
    });
    expect(table.header).toEqual(["name", "value"]);
    expect(table.rows).toEqual([["Acme", "12"]]);
  });

  it("reads a file with more rows than a call takes arguments", () => {
    const text = ["id", ...Array.from({ length: 200_000 }, () => "1")].join("\n");
    expect(tableFromText(text, { name: "ids.csv" }).totalRows).toBe(200_000);
  });
});

describe("formatJson", () => {
  it("lays JSON out and leaves anything else alone", () => {
    expect(formatJson('{"a":[1,2]}')).toBe('{\n  "a": [\n    1,\n    2\n  ]\n}');
    expect(formatJson("not json")).toBe("not json");
  });

  it("keeps the file's own numbers, keys and escapes", () => {
    expect(formatJson('{"id": 12345678901234567890, "a": 1, "a": "\\u00e9\\"}"}')).toBe(
      '{\n  "id": 12345678901234567890,\n  "a": 1,\n  "a": "\\u00e9\\"}"\n}',
    );
  });

  it("keeps empty objects and lists on one line", () => {
    expect(formatJson('{"a": {}, "b": [ ]}')).toBe('{\n  "a": {},\n  "b": []\n}');
  });

  it("leaves deeply nested JSON as written rather than ballooning it", () => {
    const nested = "[".repeat(10_000) + "0" + "]".repeat(10_000);
    expect(formatJson(nested)).toBe(nested);
  });
});

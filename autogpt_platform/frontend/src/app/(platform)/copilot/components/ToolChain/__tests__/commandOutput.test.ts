import { describe, expect, it } from "vitest";
import {
  dataFile,
  describeData,
  previewColumns,
  summarizeData,
} from "../commandOutput";
import { stepFailed } from "../stepOutcome";

describe("summarizeData", () => {
  it("reads CSV with quoted commas as one table", () => {
    const csv = 'name,note\n"Rs 899, 30ml",launch\nserum,"vitamin C, 15%"';
    expect(summarizeData(csv)).toEqual({
      kind: "table",
      rows: 2,
      columns: ["name", "note"],
      delimiter: ",",
      sample: [
        ["Rs 899, 30ml", "launch"],
        ["serum", "vitamin C, 15%"],
      ],
    });
  });

  it("reads TSV", () => {
    expect(
      summarizeData("date\tpost\n2026-11-25\tLaunch\n2026-11-28\tReviews"),
    ).toMatchObject({
      kind: "table",
      rows: 2,
      columns: ["date", "post"],
      delimiter: "\t",
    });
  });

  it("leaves two or three lines of prose with a comma each alone", () => {
    expect(summarizeData("Hello there, friend\nSee you soon, bye")).toBeNull();
    expect(
      summarizeData(
        "Updated the brief, as asked.\nChecked the dates, all fine.",
      ),
    ).toBeNull();
  });

  it("leaves prose, a single line and ragged rows alone", () => {
    expect(
      summarizeData("Done. Updated the brief, then the calendar.\nAll good."),
    ).toBeNull();
    expect(summarizeData("a,b,c")).toBeNull();
    expect(summarizeData("a,b\n1,2\n1,2,3")).toBeNull();
    expect(summarizeData("   ")).toBeNull();
    expect(
      summarizeData(
        "This sentence is far too long to be a column name, honestly\nx,y",
      ),
    ).toBeNull();
  });

  it("reads JSON objects and lists, and ignores what only looks like JSON", () => {
    expect(
      summarizeData('{"a": 1, "b": "two", "c": {"nested": true}, "~/x/y": 3}'),
    ).toEqual({
      kind: "json",
      fields: ["a", "b", "c", "~/x/y"],
      sample: [
        ["a", "1"],
        ["b", "two"],
      ],
    });
    expect(summarizeData("[1, 2, 3]")).toEqual({ kind: "json-list", items: 3 });
    expect(summarizeData("{not json")).toBeNull();
  });
});

describe("describeData", () => {
  it("names a table by its rows and first columns", () => {
    expect(
      describeData({
        kind: "table",
        rows: 1,
        columns: ["a", "b", "c", "d", "e"],
        delimiter: ",",
        sample: [],
      }),
    ).toEqual({ title: "Table", detail: "1 row · a, b, c, d, …" });
  });

  it("never previews a path or other technical name, only counts it", () => {
    expect(
      describeData({
        kind: "json",
        fields: ["~/workspace/dewleaf/brief.md", "launch"],
        sample: [],
      }),
    ).toEqual({ title: "Data", detail: "2 fields · launch" });
    expect(
      describeData({
        kind: "table",
        rows: 3,
        columns: ["src/a.py", "b.json"],
        delimiter: ",",
        sample: [],
      }),
    ).toEqual({ title: "Table", detail: "3 rows · 2 columns" });
  });

  it("names JSON by its fields or items", () => {
    expect(
      describeData({ kind: "json", fields: ["launch"], sample: [] }),
    ).toEqual({
      title: "Data",
      detail: "1 field · launch",
    });
    expect(describeData({ kind: "json", fields: [], sample: [] }).detail).toBe(
      "No fields",
    );
    expect(describeData({ kind: "json-list", items: 2 })).toEqual({
      title: "Data",
      detail: "2 items",
    });
  });
});

describe("stepFailed", () => {
  const exit = (code: number) => ({ stdout: "", stderr: "", exit_code: code });

  it("is a failure for any non-zero exit by default", () => {
    expect(stepFailed("curl -fs https://x", exit(22))).toBe(true);
    expect(stepFailed("./report.sh", exit(1))).toBe(true);
    expect(stepFailed(null, exit(1))).toBe(true);
  });

  it("reads exit 1 with nothing on stderr from a test-style command as a 'no'", () => {
    expect(stepFailed("grep -q TBD brief.md", exit(1))).toBe(false);
    expect(stepFailed("cd drafts; test -f brief.md", exit(1))).toBe(false);
    expect(stepFailed("ls drafts\ntest -f brief.md", exit(1))).toBe(false);
    expect(stepFailed("! [ -d drafts ]", exit(1))).toBe(false);
    expect(stepFailed("grep -q TBD brief.md", exit(2))).toBe(true);
  });

  it("reads a test-style && or || chain that exits 1 quietly as a 'no'", () => {
    expect(stepFailed("grep -q TBD brief.md && echo found", exit(1))).toBe(
      false,
    );
    expect(stepFailed("cd drafts && test -f brief.md", exit(1))).toBe(false);
    expect(stepFailed("test -f a || grep -q x b", exit(1))).toBe(false);
    expect(stepFailed("grep -q x a && wc -l a", exit(1))).toBe(false);
    expect(stepFailed("grep -q x f && echo ok", exit(1))).toBe(false);
    expect(stepFailed("./build.sh || grep -q x f", exit(1))).toBe(false);
    expect(stepFailed("cat log | grep -q x && echo ok", exit(1))).toBe(false);
  });

  it("calls a quiet exit 1 a failure when a command that is not a test could have produced it", () => {
    expect(stepFailed("grep -q x f && ./publish.sh", exit(1))).toBe(true);
    expect(stepFailed("grep -q x f || ./fallback.sh", exit(1))).toBe(true);
    expect(
      stepFailed("test -f a && ./publish.sh || grep -q x b", exit(1)),
    ).toBe(false);
    expect(stepFailed("grep -q x f && ./publish.sh && echo ok", exit(1))).toBe(
      true,
    );
    expect(stepFailed("grep -q x f | ./publish.sh", exit(1))).toBe(true);
    expect(stepFailed("echo ok && ./publish.sh", exit(1))).toBe(true);
    expect(stepFailed("echo ok", exit(1))).toBe(true);
  });

  it("calls a step that complained on stderr a failure, test-style or not", () => {
    const complained = (stderr: string, code = 1) => ({
      stdout: "",
      stderr,
      exit_code: code,
    });
    expect(
      stepFailed(
        "cd missing && grep -q TBD brief.md",
        complained("cd: missing: No such file or directory"),
      ),
    ).toBe(true);
    expect(
      stepFailed(
        "grep -q TBD gone.md",
        complained("grep: gone.md: No such file", 2),
      ),
    ).toBe(true);
  });

  it("reads a quoted | or ; as part of the argument, not a new command", () => {
    expect(stepFailed('grep -Eq "TBD|TODO" brief.md', exit(1))).toBe(false);
    expect(stepFailed("grep -q 'a;b' notes.txt", exit(1))).toBe(false);
    expect(stepFailed('cat log | grep -q "x|y"', exit(1))).toBe(false);
    expect(stepFailed('grep -q "a|b" f; ./publish.sh', exit(1))).toBe(true);
    expect(stepFailed("grep -q TBD brief.md 2>&1", exit(1))).toBe(false);
    expect(stepFailed("grep -q x\\;y notes.txt", exit(1))).toBe(false);
    expect(stepFailed("sleep 5 & ./sync.sh", exit(1))).toBe(true);
  });

  it("is a failure for a timeout or a refusal, and fine for exit 0", () => {
    expect(stepFailed("sleep 9", { timed_out: true, exit_code: -1 })).toBe(
      true,
    );
    expect(stepFailed("", { type: "error", message: "No command" })).toBe(true);
    expect(stepFailed("ls", exit(0))).toBe(false);
    expect(stepFailed("ls", null)).toBe(false);
  });
});

describe("dataFile and previewColumns", () => {
  it("names the download by what was printed", () => {
    const table = {
      kind: "table" as const,
      rows: 1,
      columns: ["a", "b"],
      sample: [],
    };
    expect(dataFile({ ...table, delimiter: "," }).name).toBe("table.csv");
    expect(dataFile({ ...table, delimiter: "\t" }).name).toBe("table.tsv");
    expect(dataFile({ kind: "json-list", items: 2 }).label).toBe(
      "Download JSON",
    );
  });

  it("leaves out a column whose sampled values are technical", () => {
    expect(
      previewColumns(
        ["date", "file", "status"],
        [
          ["2026-11-10", "~/workspace/a/brief.md", "done"],
          ["2026-11-13", "notes", "draft"],
        ],
      ),
    ).toEqual([0, 2]);
  });

  it("previews the first four plain columns", () => {
    expect(previewColumns(["~/x/y", "a", "b", "c", "d", "e"])).toEqual([
      1, 2, 3, 4,
    ]);
  });
});

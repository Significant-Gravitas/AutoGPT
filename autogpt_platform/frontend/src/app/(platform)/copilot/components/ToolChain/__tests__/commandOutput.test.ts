import { describe, expect, it } from "vitest";
import {
  dataFile,
  describeData,
  previewColumns,
  summarizeData,
} from "../commandOutput";
import { stepFailed, stepOutput } from "../stepOutcome";

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
    expect(stepFailed("cat log | grep -q x && echo ok", exit(1))).toBe(false);
  });

  it("calls a quiet exit 1 a failure when a command that is not a test could have produced it", () => {
    expect(stepFailed("grep -q x f && ./publish.sh", exit(1))).toBe(true);
    expect(stepFailed("grep -q x f || ./fallback.sh", exit(1))).toBe(true);
    expect(
      stepFailed("test -f a && ./publish.sh || grep -q x b", exit(1)),
    ).toBe(true);
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

describe("stepFailed, from what ran", () => {
  const exit = (code: number) => ({ stdout: "", stderr: "", exit_code: code });

  it("calls exit 1 a failure when a command must have failed on the way to it", () => {
    // `||` only runs grep after build.sh failed: both failed, quietly.
    expect(stepFailed("./build.sh || grep -q x f", exit(1))).toBe(true);
    expect(stepFailed("./a.sh || ./b.sh || grep -q x f", exit(1))).toBe(true);
    expect(
      stepFailed("grep -q x f && ./publish.sh || test -f a", exit(1)),
    ).toBe(true);
  });

  it("calls exit 1 a failure when a silenced command could have set it", () => {
    expect(
      stepFailed("cd drafts 2>/dev/null && test -f brief.md", exit(1)),
    ).toBe(true);
    expect(stepFailed("cd drafts &>/dev/null && grep -q x f", exit(1))).toBe(
      true,
    );
    expect(stepFailed("mkdir out 2>&1 && test -d out/a", exit(1))).toBe(true);
  });

  it("still reads a chain of tests, or a loud command then a test, as a 'no'", () => {
    expect(stepFailed("test -f a || grep -q x b", exit(1))).toBe(false);
    expect(stepFailed("[ -d d ] || test -f a || grep -q x b", exit(1))).toBe(
      false,
    );
    expect(stepFailed("cd drafts && test -f brief.md", exit(1))).toBe(false);
    expect(stepFailed("grep -q x f 2>/dev/null", exit(1))).toBe(false);
    expect(stepFailed("grep -q x f 2>/dev/null || grep -q y f", exit(1))).toBe(
      false,
    );
    expect(stepFailed("! test -f a || grep -q x f", exit(1))).toBe(false);
    expect(stepFailed("cd drafts >&2 && test -f brief.md", exit(1))).toBe(
      false,
    );
  });
});

describe("stepFailed, pipelines and shell options", () => {
  const exit = (code: number) => ({ stdout: "", stderr: "", exit_code: code });

  it("calls exit 1 a failure when an earlier stage of a pipeline could have failed", () => {
    // Under pipefail the status is the rightmost failing stage: build.sh's own.
    expect(stepFailed("set -o pipefail; ./build.sh | grep -q x", exit(1))).toBe(
      true,
    );
    expect(
      stepFailed("set -euo pipefail\n./build.sh | grep -q x", exit(1)),
    ).toBe(true);
    // Without it, a build that failed before printing leaves grep nothing to find.
    expect(stepFailed("./build.sh | grep -q x", exit(1))).toBe(true);
    expect(
      stepFailed(
        "curl -s https://shop.example | grep -q ok && echo up",
        exit(1),
      ),
    ).toBe(true);
    expect(stepFailed("cat log 2>/dev/null | grep -q x", exit(1))).toBe(true);
  });

  it("calls exit 1 a failure when set -e let an earlier command end the run", () => {
    expect(stepFailed("set -e; ./build.sh; grep -q x f", exit(1))).toBe(true);
    expect(stepFailed("set -o errexit\n./build.sh\ntest -f out", exit(1))).toBe(
      true,
    );
  });

  it("still reads tests fed by commands that cannot fail quietly as a 'no'", () => {
    expect(stepFailed("cat log | grep -q x", exit(1))).toBe(false);
    expect(stepFailed("set -o pipefail; cat log | grep -q x", exit(1))).toBe(
      false,
    );
    expect(stepFailed("grep x f | grep -q y", exit(1))).toBe(false);
    expect(stepFailed("! cat log | grep -q x || test -f a", exit(1))).toBe(
      false,
    );
    expect(stepFailed("set -e; cd drafts; test -f brief.md", exit(1))).toBe(
      false,
    );
    expect(stepFailed("set -e; set +e; ./build.sh; grep -q x f", exit(1))).toBe(
      false,
    );
    expect(stepFailed("./build.sh; grep -q x f", exit(1))).toBe(false);
  });
});

describe("stepOutput", () => {
  it("reads the outcome of a result too large for the chat from its kept tail", () => {
    const failed = truncatedEnvelope({
      stdout: "x".repeat(120_000),
      stderr: "",
      exit_code: 3,
    });
    expect(stepOutput(failed)).toMatchObject({ stderr: "", exit_code: 3 });
    expect(stepFailed("./export.sh", stepOutput(failed))).toBe(true);

    const fine = truncatedEnvelope({
      stdout: "x".repeat(120_000),
      stderr: "",
      exit_code: 0,
    });
    expect(stepFailed("./export.sh", stepOutput(fine))).toBe(false);
  });

  it("reads a timeout or a complaint on stderr from the tail too", () => {
    const timedOut = truncatedEnvelope({
      stdout: "x".repeat(120_000),
      stderr: "",
      exit_code: -1,
      timed_out: true,
    });
    expect(stepOutput(timedOut)).toMatchObject({ timed_out: true });
    const complained = truncatedEnvelope({
      stdout: "x".repeat(120_000),
      stderr: 'cut: "x": bad delimiter',
      exit_code: 1,
    });
    expect(stepOutput(complained)).toMatchObject({
      stderr: 'cut: "x": bad delimiter',
    });
    expect(stepFailed("cut -d x f", stepOutput(complained))).toBe(true);
  });

  it("ignores an exit code printed inside the output itself", () => {
    const printed = truncatedEnvelope({
      stdout: `{"exit_code": 0}${"x".repeat(120_000)}`,
      stderr: "",
      exit_code: 2,
    });
    expect(stepOutput(printed)).toMatchObject({ exit_code: 2 });
  });

  it("skips a field it cannot read rather than throwing", () => {
    const envelope =
      '<tool-output-truncated total_chars=9>\n{"stdout": "x… (omitted 9 chars)…y", "stderr": "bad \\q", "exit_code": 4}\n</tool-output-truncated>';
    expect(() => stepOutput(envelope)).not.toThrow();
    expect(stepOutput(envelope)).toEqual({ exit_code: 4 });
  });

  it("passes an object through and reads nothing from other text", () => {
    expect(stepOutput({ exit_code: 0 })).toEqual({ exit_code: 0 });
    expect(stepOutput("plain text")).toBeNull();
    expect(stepOutput(null)).toBeNull();
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

/** A bash result as the backend sends one too large for the chat: the JSON,
 *  cut in the middle, inside a `<tool-output-truncated>` envelope. */
function truncatedEnvelope(fields: Record<string, unknown>): string {
  const json = JSON.stringify({
    type: "bash_exec",
    message: "Command executed",
    session_id: "s",
    ...fields,
    timed_out: fields.timed_out ?? false,
  });
  const kept = 2_000;
  const cut = `${json.slice(0, kept)}… (omitted ${json.length - 2 * kept} chars)…${json.slice(-kept)}`;
  return `<tool-output-truncated total_chars=${json.length} workspace_path="tool-outputs/c.json">\n${cut}\nFull output (${json.length} chars) saved to workspace.\n</tool-output-truncated>`;
}

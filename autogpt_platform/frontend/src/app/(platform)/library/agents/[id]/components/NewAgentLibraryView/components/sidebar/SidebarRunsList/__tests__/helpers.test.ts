import { describe, expect, it } from "vitest";
import { uniqueRunsByID } from "../helpers";

describe("uniqueRunsByID", () => {
  it("keeps the first copy of a run that two pages both returned", () => {
    const first = { id: "b", page: 1 };
    const runs = [{ id: "a" }, first, { id: "b", page: 2 }, { id: "c" }];

    const unique = uniqueRunsByID(runs);

    expect(unique.map((run) => run.id)).toEqual(["a", "b", "c"]);
    expect(unique[1]).toBe(first);
  });

  it("returns every run when none repeat", () => {
    const runs = [{ id: "a" }, { id: "b" }];
    expect(uniqueRunsByID(runs)).toEqual(runs);
  });
});

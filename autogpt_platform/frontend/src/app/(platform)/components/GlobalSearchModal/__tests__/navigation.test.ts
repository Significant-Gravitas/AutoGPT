import { describe, expect, it } from "vitest";
import { buildNavigationBucket } from "../navigation";

function navIDs(query: string) {
  return (
    buildNavigationBucket(query, null).bucket?.items.map((item) => item.id) ??
    []
  );
}

describe("buildNavigationBucket", () => {
  it.each(["workflows", "my workflows", "agents", "my agents"])(
    "finds the Library when searching %s",
    (query) => {
      expect(navIDs(query)).toContain("nav:library");
    },
  );
});

import { describe, expect, test } from "vitest";
import { queryToAsyncStatus } from "./async-status";

describe("queryToAsyncStatus", () => {
  test("settles disabled queries", () => {
    expect(
      queryToAsyncStatus(
        {
          data: undefined,
          enabled: false,
          isError: false,
          isFetching: false,
        },
        { keepCachedData: false },
      ),
    ).toBe("loaded");
  });

  test("applies the cached-data policy explicitly", () => {
    const query = {
      data: ["cached"],
      enabled: true,
      isError: true,
      isFetching: false,
    };

    expect(queryToAsyncStatus(query, { keepCachedData: true })).toBe("loaded");
    expect(queryToAsyncStatus(query, { keepCachedData: false })).toBe("error");
  });

  test("distinguishes loading, failure, and loaded data", () => {
    expect(
      queryToAsyncStatus(
        {
          data: undefined,
          enabled: true,
          isError: true,
          isFetching: true,
        },
        { keepCachedData: false },
      ),
    ).toBe("loading");
    expect(
      queryToAsyncStatus(
        {
          data: undefined,
          enabled: true,
          isError: true,
          isFetching: false,
        },
        { keepCachedData: false },
      ),
    ).toBe("error");
    expect(
      queryToAsyncStatus(
        {
          data: [],
          enabled: true,
          isError: false,
          isFetching: false,
        },
        { keepCachedData: false },
      ),
    ).toBe("loaded");
  });
});

import { describe, expect, it } from "vitest";
import {
  filterSessions,
  formatRelativeDate,
  highlightMatch,
  type SearchSession,
} from "../helpers";

const sessions: SearchSession[] = [
  {
    id: "older",
    title: "Weekly update",
    updated_at: "2025-01-01T00:00:00Z",
  },
  {
    id: "newer",
    title: "Revenue forecast",
    updated_at: "2025-01-03T00:00:00Z",
  },
  {
    id: "middle",
    title: "Forecast follow-up",
    updated_at: "2025-01-02T00:00:00Z",
  },
];

describe("filterSessions", () => {
  it("returns recent sessions sorted newest first for an empty query", () => {
    expect(filterSessions(sessions, "").map((session) => session.id)).toEqual([
      "newer",
      "middle",
      "older",
    ]);
  });

  it("filters titles case-insensitively and keeps newest first", () => {
    expect(
      filterSessions(sessions, "FORECAST").map((session) => session.id),
    ).toEqual(["newer", "middle"]);
  });

  it("limits active search results to 20 sessions", () => {
    const manySessions = Array.from({ length: 25 }, (_, index) => ({
      id: String(index),
      title: `Searchable ${index}`,
      updated_at: `2025-01-${String(index + 1).padStart(2, "0")}T00:00:00Z`,
    }));

    expect(filterSessions(manySessions, "searchable")).toHaveLength(20);
  });
});

describe("highlightMatch", () => {
  it("marks the matched title substring", () => {
    expect(highlightMatch("Revenue forecast", "fore")).toEqual([
      { text: "Revenue ", isMatch: false },
      { text: "fore", isMatch: true },
      { text: "cast", isMatch: false },
    ]);
  });

  it("returns plain text when there is no query", () => {
    expect(highlightMatch("Revenue forecast", "")).toEqual([
      { text: "Revenue forecast", isMatch: false },
    ]);
  });
});

describe("formatRelativeDate", () => {
  const baseDate = new Date("2025-01-10T12:00:00Z");

  it("formats recent days", () => {
    expect(formatRelativeDate("2025-01-10T00:00:00Z", baseDate)).toBe("Today");
    expect(formatRelativeDate("2025-01-09T00:00:00Z", baseDate)).toBe(
      "Yesterday",
    );
    expect(formatRelativeDate("2025-01-07T00:00:00Z", baseDate)).toBe(
      "3 days ago",
    );
  });

  it("formats older dates", () => {
    expect(formatRelativeDate("2025-01-01T00:00:00Z", baseDate)).toBe(
      "1st Jan 2025",
    );
  });
});

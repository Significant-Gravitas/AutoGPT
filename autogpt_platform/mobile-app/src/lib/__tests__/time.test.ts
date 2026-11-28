import { shortTime } from "../time";

// Wednesday 11 November 2026, 3pm local time.
const now = new Date(2026, 10, 11, 15, 0);

function iso(year: number, month: number, day: number, hour = 9, minute = 30) {
  return new Date(year, month, day, hour, minute).toISOString();
}

describe("shortTime", () => {
  it("shows the time for a chat from today", () => {
    const label = shortTime(iso(2026, 10, 11, 9, 5), now);
    expect(label).toMatch(/9/);
    expect(label).toMatch(/05/);
    expect(label).not.toBe("Yesterday");
  });

  it("says Yesterday for the day before", () => {
    expect(shortTime(iso(2026, 10, 10, 23, 59), now)).toBe("Yesterday");
  });

  it("shows a short weekday within the last week", () => {
    const label = shortTime(iso(2026, 10, 9), now);
    expect(label).toBe(new Date(2026, 10, 9).toLocaleDateString([], { weekday: "short" }));
    expect(label).not.toMatch(/\d/);
  });

  it("shows the day and short month for older chats", () => {
    const label = shortTime(iso(2026, 2, 6), now);
    expect(label).toMatch(/^6 /);
    expect(label).toContain(new Date(2026, 2, 6).toLocaleDateString([], { month: "short" }));
  });

  it("returns null for a date it can't read", () => {
    expect(shortTime("not a date", now)).toBeNull();
  });
});

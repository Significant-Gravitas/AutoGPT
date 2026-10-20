import { describe, expect, it } from "vitest";
import type { AccuracyLatestData } from "@/app/api/__generated__/models/accuracyLatestData";
import { toAccuracyChartData } from "../accuracyChartData";

function point(date: string, score: number): AccuracyLatestData {
  return {
    date: date as unknown as Date,
    daily_score: score,
    three_day_avg: score + 1,
    seven_day_avg: score + 2,
    fourteen_day_avg: score + 3,
  };
}

const label = (date: string) => new Date(date).toLocaleDateString();

describe("toAccuracyChartData", () => {
  it("puts the oldest date first when the API returns newest first", () => {
    const newestFirst = [
      point("2026-10-16T00:00:00Z", 50),
      point("2026-10-15T00:00:00Z", 60),
      point("2026-10-14T00:00:00Z", 70),
    ];

    const chart = toAccuracyChartData(newestFirst);

    expect(chart.map((p) => p.date)).toEqual([
      label("2026-10-14T00:00:00Z"),
      label("2026-10-15T00:00:00Z"),
      label("2026-10-16T00:00:00Z"),
    ]);
    expect(chart.map((p) => p["Daily Score"])).toEqual([70, 60, 50]);
    expect(chart.map((p) => p["3-Day Avg"])).toEqual([71, 61, 51]);
    expect(chart.map((p) => p["7-Day Avg"])).toEqual([72, 62, 52]);
    expect(chart.map((p) => p["14-Day Avg"])).toEqual([73, 63, 53]);
  });

  it("sorts unordered input and keeps already-ascending input as is", () => {
    const shuffled = [
      point("2026-10-15T00:00:00Z", 60),
      point("2026-10-16T00:00:00Z", 50),
      point("2026-10-14T00:00:00Z", 70),
    ];
    const ascending = [
      point("2026-10-14T00:00:00Z", 70),
      point("2026-10-15T00:00:00Z", 60),
      point("2026-10-16T00:00:00Z", 50),
    ];

    expect(toAccuracyChartData(shuffled)).toEqual(
      toAccuracyChartData(ascending),
    );
    expect(toAccuracyChartData(ascending)[0].date).toBe(
      label("2026-10-14T00:00:00Z"),
    );
  });

  it("does not mutate the input array", () => {
    const input = [
      point("2026-10-16T00:00:00Z", 50),
      point("2026-10-14T00:00:00Z", 70),
    ];
    const before = [...input];

    toAccuracyChartData(input);

    expect(input).toEqual(before);
  });

  it("preserves the calendar date of the API's timezone-free daily buckets", () => {
    const chart = toAccuracyChartData([point("2026-10-16T00:00:00", 50)]);

    expect(chart[0].date).toBe(new Date(2026, 9, 16).toLocaleDateString());
  });
});

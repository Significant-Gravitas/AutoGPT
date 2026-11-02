import type { AccuracyLatestData } from "@/app/api/__generated__/models/accuracyLatestData";

export interface AccuracyChartPoint {
  date: string;
  "Daily Score": AccuracyLatestData["daily_score"];
  "3-Day Avg": AccuracyLatestData["three_day_avg"];
  "7-Day Avg": AccuracyLatestData["seven_day_avg"];
  "14-Day Avg": AccuracyLatestData["fourteen_day_avg"];
}

/**
 * Map accuracy history to chart points, oldest date first.
 *
 * Recharts plots a category X axis in array order, so sort a copy by date
 * rather than trusting the order the API returns.
 */
export function toAccuracyChartData(
  data: AccuracyLatestData[],
): AccuracyChartPoint[] {
  return [...data]
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    .map((item) => ({
      date: new Date(item.date).toLocaleDateString(),
      "Daily Score": item.daily_score,
      "3-Day Avg": item.three_day_avg,
      "7-Day Avg": item.seven_day_avg,
      "14-Day Avg": item.fourteen_day_avg,
    }));
}

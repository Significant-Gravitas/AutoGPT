import type { ExecutionAnalyticsResult } from "@/app/api/__generated__/models/executionAnalyticsResult";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { AnalyticsResultsTable } from "../AnalyticsResultsTable";

function result(execId: string, score: number): ExecutionAnalyticsResult {
  return {
    agent_id: "agent-1",
    version_id: 1,
    user_id: "user-1",
    exec_id: execId,
    summary_text: null,
    score,
    status: "success",
  };
}

describe("AnalyticsResultsTable", () => {
  test("shows each execution's score as a percentage", () => {
    render(
      <AnalyticsResultsTable
        results={{
          total_executions: 4,
          processed_executions: 4,
          successful_analytics: 4,
          failed_analytics: 0,
          skipped_executions: 0,
          results: [
            result("exec-high", 0.9),
            result("exec-good", 0.7),
            result("exec-fair", 0.5),
            result("exec-low", 0.2),
          ],
        }}
      />,
    );

    for (const percentage of ["90%", "70%", "50%", "20%"]) {
      expect(screen.getByText(percentage)).toBeDefined();
    }
  });
});

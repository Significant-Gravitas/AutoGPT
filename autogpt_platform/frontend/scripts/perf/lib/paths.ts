import path from "path";

export const FRONTEND_ROOT = path.resolve(__dirname, "../../..");
export const PERF_OUTPUT_DIR = path.join(FRONTEND_ROOT, ".perf");
export const FIXTURES_DIR = path.resolve(__dirname, "../fixtures");

export function scenarioOutputDir(scenario: string) {
  return path.join(PERF_OUTPUT_DIR, scenario);
}

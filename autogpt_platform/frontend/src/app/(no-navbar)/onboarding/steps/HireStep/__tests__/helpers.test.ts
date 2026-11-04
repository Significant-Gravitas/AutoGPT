import { describe, expect, it } from "vitest";
import type { ExpertRecommendations } from "@/app/api/__generated__/models/expertRecommendations";
import {
  HIRE_TITLE_EMPTY,
  HIRE_TITLE_EMPTY_NO_READ,
  hireDiagnosis,
  hireTitle,
} from "../helpers";

function team(diagnosis: string): ExpertRecommendations {
  return { experts: [], diagnosis } as unknown as ExpertRecommendations;
}

describe("hireTitle", () => {
  it("promises a read only when there is one to show", () => {
    expect(hireTitle(team("You need help with support."), false)).toBe(
      HIRE_TITLE_EMPTY,
    );
    expect(hireDiagnosis(team("You need help with support."), false)).toBe(
      "You need help with support.",
    );
  });

  it("never titles a blank read as one", () => {
    expect(hireTitle(team("   "), false)).toBe(HIRE_TITLE_EMPTY_NO_READ);
    expect(hireDiagnosis(team("   "), false)).toBeNull();
  });
});

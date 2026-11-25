import { accessProviders, hirePerks, planLine, profileStats } from "../profileHelpers";

function workflow(providers: string[]) {
  return { id: providers.join(), name: null, description: null, schedule_cron: null, integration_providers: providers };
}

describe("profileStats", () => {
  it("counts skills, workflows and services, leaving out the zeros", () => {
    const skill = { id: "s", slug: "s", title: "S", description: "" };
    expect(profileStats({ bundled_skills: [skill], workflows: [workflow([]), workflow([])] }, 3)).toEqual([
      "1 skill",
      "2 workflows",
      "3 services",
    ]);
  });

  it("shows nothing for an expert with none of them, or a list cached without workflows", () => {
    expect(profileStats({ bundled_skills: [], workflows: undefined }, 0)).toEqual([]);
  });
});

describe("accessProviders", () => {
  it("names each service once and drops the ones the platform pays for", () => {
    const workflows = [workflow(["google", "openai"]), workflow(["google", "notion"])];
    expect(accessProviders(workflows, ["openai"])).toEqual(["google", "notion"]);
  });

  it("names none until the platform's list is in", () => {
    expect(accessProviders([workflow(["google"])], undefined)).toEqual([]);
  });
});

describe("hirePerks", () => {
  it("mentions the workflows only when there are some", () => {
    expect(hirePerks(1)).toContain("1 workflow installed and ready to run");
    expect(hirePerks(0)).toHaveLength(3);
  });
});

describe("planLine", () => {
  it("states the weekly cap in dollars", () => {
    expect(planLine("Remy", 500)).toBe(
      "Hiring Remy costs nothing extra. Their work draws on your existing balance, capped at $5 a week — a limit you can change once they are on your team.",
    );
    expect(planLine("Remy", 250)).toContain("$2.50 a week");
  });

  it("ends at the balance without a cap", () => {
    expect(planLine("Remy", null)).toBe(
      "Hiring Remy costs nothing extra. Their work draws on your existing balance.",
    );
  });
});

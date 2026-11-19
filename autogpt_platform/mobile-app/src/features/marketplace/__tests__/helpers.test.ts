import type { Expert, ExpertTemplate } from "../../../lib/api/experts";
import {
  emptyMessage,
  filterTemplates,
  hireErrorMessage,
  hiredTemplateIDs,
  withHiredExpert,
} from "../helpers";

function expert(overrides: Partial<Expert>): Expert {
  return {
    id: "e1",
    name: "Alex",
    avatar_url: null,
    role: "Product",
    job_title: null,
    tagline: null,
    categories: [],
    last_run_at: null,
    last_run_status: null,
    schedules_paused_at: null,
    setup_status: "ready",
    source_template_id: null,
    ...overrides,
  };
}

function template(overrides: Partial<ExpertTemplate>): ExpertTemplate {
  return { ...expert(overrides), bio: null, bundled_skills: [], ...overrides };
}

const alex = template({
  id: "t-alex",
  name: "Alex",
  role: "Product",
  job_title: "Product Manager",
  tagline: "Scores the backlog, writes the spec.",
  categories: ["development", "research"],
});
const kai = template({
  id: "t-kai",
  name: "Kai",
  role: "Executive Assistant",
  job_title: "Executive Assistant",
  tagline: "Sorts your inbox.",
  categories: ["support", "operations"],
});

describe("filterTemplates", () => {
  it("returns everything for a blank query and no area", () => {
    expect(filterTemplates([alex, kai], "   ", null)).toEqual([alex, kai]);
  });

  it("matches name, job title, role and tagline, any case", () => {
    expect(filterTemplates([alex, kai], "BACKLOG", null)).toEqual([alex]);
    expect(filterTemplates([alex, kai], "assistant", null)).toEqual([kai]);
    expect(filterTemplates([alex, kai], " kai ", null)).toEqual([kai]);
  });

  it("narrows by area and query together", () => {
    expect(filterTemplates([alex, kai], "", "operations")).toEqual([kai]);
    expect(filterTemplates([alex, kai], "alex", "operations")).toEqual([]);
  });
});

describe("hiredTemplateIDs", () => {
  it("collects source template ids, skipping nulls", () => {
    expect(
      hiredTemplateIDs([{ source_template_id: "t1" }, { source_template_id: null }]),
    ).toEqual(new Set(["t1"]));
  });
});

describe("withHiredExpert", () => {
  const first = expert({ id: "a" });
  const second = expert({ id: "b" });

  it("appends a new expert", () => {
    expect(withHiredExpert([first], second)).toEqual([first, second]);
  });

  it("replaces one already there", () => {
    const renamed = { ...first, name: "New" };
    expect(withHiredExpert([first], renamed)).toEqual([renamed]);
  });

  it("starts a list when the cache is empty", () => {
    expect(withHiredExpert(undefined, first)).toEqual([first]);
  });
});

describe("hireErrorMessage", () => {
  it("says the team is full", () => {
    expect(
      hireErrorMessage({ status: 409, code: "active_expert_limit", limit: 5 }, "Alex"),
    ).toBe("Your team is full (5 experts). Remove one on the web to add another.");
  });

  it("asks to try again otherwise", () => {
    expect(hireErrorMessage({ status: 503 }, "Alex")).toBe(
      "Couldn't hire Alex. Try again.",
    );
    expect(hireErrorMessage(new Error("offline"), "Alex")).toBe(
      "Couldn't hire Alex. Try again.",
    );
  });
});

describe("emptyMessage", () => {
  it("quotes the search", () => {
    expect(emptyMessage(" tax lawyer ", null)).toEqual({
      title: "No experts match “tax lawyer”",
      detail: "Try another word or area.",
    });
  });

  it("names the area when there is no search", () => {
    expect(emptyMessage("", "content")).toEqual({
      title: "No Content experts yet",
      detail: "Try another area.",
    });
  });
});

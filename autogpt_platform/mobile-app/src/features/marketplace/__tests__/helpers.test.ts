import type { Expert, ExpertTemplate } from "../../../lib/api/experts";
import {
  areaSections,
  filterTemplates,
  firstName,
  hireErrorMessage,
  hiredTeam,
  hiredTemplateIDs,
  openSeats,
  primaryArea,
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
  it("returns everything for a blank query", () => {
    expect(filterTemplates([alex, kai], "   ")).toEqual([alex, kai]);
  });

  it("matches job title, role, name and tagline, any case", () => {
    expect(filterTemplates([alex, kai], "product man")).toEqual([alex]);
    expect(filterTemplates([alex, kai], "BACKLOG")).toEqual([alex]);
    expect(filterTemplates([alex, kai], "assistant")).toEqual([kai]);
    expect(filterTemplates([alex, kai], " kai ")).toEqual([kai]);
    expect(filterTemplates([alex, kai], "lawyer")).toEqual([]);
  });

  it("finds by role when there is no job title", () => {
    const ops = template({ id: "t-ops", name: "Ines", role: "Ops lead", job_title: null });
    expect(filterTemplates([alex, ops], "ops")).toEqual([ops]);
  });
});

describe("primaryArea", () => {
  it("takes the first category the marketplace knows, any case", () => {
    expect(primaryArea(alex)).toBe("development");
    expect(primaryArea(template({ categories: ["General", "Sales"] }))).toBe("sales");
    expect(primaryArea(template({ categories: [] }))).toBeNull();
  });
});

describe("areaSections", () => {
  const jordan = template({ id: "t-jordan", name: "Jordan", categories: ["sales"] });
  const maria = template({ id: "t-maria", name: "Maria", categories: ["marketing"] });
  const noor = template({ id: "t-noor", name: "Noor", categories: ["marketing", "sales"] });

  it("lists each template under its first area only, skipping empty areas", () => {
    const sections = areaSections([jordan, maria, noor], new Set());
    expect(sections.map((section) => [section.area, section.templates])).toEqual([
      ["marketing", [maria, noor]],
      ["sales", [jordan]],
    ]);
  });

  it("lists templates in no known area last, never as an open seat", () => {
    const loner = template({ id: "t-loner", name: "Loner", categories: [] });
    const odd = template({ id: "t-odd", name: "Odd", categories: ["astrology"] });
    const sections = areaSections([loner, jordan, odd], new Set());
    expect(sections.map((section) => [section.area, section.templates])).toEqual([
      ["sales", [jordan]],
      [null, [loner, odd]],
    ]);
    expect(openSeats(sections)).toEqual([{ area: "sales", label: "Sales" }]);
  });

  it("puts areas nobody on the team covers first, each part in area order", () => {
    const sections = areaSections([jordan, maria, noor, alex, kai], new Set(["t-maria", "t-kai"]));
    expect(sections.map((section) => [section.area, section.isCovered])).toEqual([
      ["sales", false],
      ["development", false],
      ["marketing", true],
      ["support", true],
    ]);
  });
});

describe("firstName", () => {
  it("takes the first word", () => {
    expect(firstName(" Maria Lopez ")).toBe("Maria");
  });
});

describe("hiredTeam", () => {
  const hired = expert({ id: "e-maria", source_template_id: "t-maria" });
  const created = expert({ id: "e-own", source_template_id: null });

  it("keeps the experts hired from the marketplace", () => {
    expect(hiredTeam([hired, created], new Map())).toEqual([hired]);
  });

  it("adds a hire the list doesn't have yet, once, with its template", () => {
    const fresh = expert({ id: "e-jordan", source_template_id: null });
    const team = hiredTeam([hired], new Map([["t-jordan", fresh], ["t-maria", hired]]));
    expect(team).toEqual([hired, { ...fresh, source_template_id: "t-jordan" }]);
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

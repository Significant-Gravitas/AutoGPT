import { describe, expect, it } from "vitest";
import { parseWorkspace, workspaceText } from "../parse";
import { taskErrors, type BoardTask } from "../catalog-tasks";
import { localInstants, resolveDateTime } from "../temporal";
import { planItinerary } from "../itinerary";
import { itineraryCalendar } from "../calendar-export";
import { csvText } from "../download";
import {
  dataDashboard,
  formView,
  tripProps,
  tripSource,
} from "./capability-fixtures";

describe("component contracts and data integrity", () => {
  it("validates connected data and produces a matching summary", () => {
    expect(parseWorkspace(dataDashboard())).toBeDefined();
    const text = workspaceText(dataDashboard());
    expect(text).toContain("Total revenue: 300 USD");
    expect(text).toContain("Total cost: Unknown USD");
  });
  it("rejects disconnected dashboards and reserved field collisions", () => {
    expect(() =>
      parseWorkspace(
        dataDashboard().replace('field: "data"', 'field: "missing"'),
      ),
    ).toThrow(/binding/);
    expect(() =>
      parseWorkspace(formView('NumberField("a__draft", "Count", 1, 0, 10, 1)')),
    ).toThrow(/reserved/);
  });
  it("requires meaningful conditional field references", () => {
    expect(() =>
      parseWorkspace(
        formView(
          'FieldGroup("details", "Details", [{key:"name",label:"Name",type:"text"}], [{id:"one",cells:{name:null}}], false, true, {field:"missing",equals:true})',
        ),
      ),
    ).toThrow(/condition/);
  });
  it.each([
    "javascript:alert(1)",
    "workspace://../other#image/png",
    "workspace://file?url=https://example.org",
    "data:image/svg+xml,<svg />",
  ])("rejects unsafe artifact paths: %s", (uri) => {
    expect(() =>
      parseWorkspace(
        `root = Workspace("Output", "", [Artifact("Image", ${JSON.stringify(uri)}, "image/png", "art.png", "Generated art")])`,
      ),
    ).toThrow();
  });
  it("keeps signed numbers and escapes spreadsheet formulas on export", () => {
    expect(
      csvText([
        ["value"],
        [-10.5],
        ['=IMPORTXML("https://example.org")'],
        ["multi\nline"],
      ]),
    ).toContain('"-10.5"');
    expect(csvText([["=1+1"]])).toContain("'=1+1");
  });
  it("retains null chart gaps and rejects incomplete/misleading series", () => {
    const source =
      'root = Workspace("Trend", "", [SeriesChart("Trend", "", "USD", "line", [{key:"a",label:"A"},{key:"b",label:"B"}], [{label:"Jan",values:{a:2,b:null}}])])';
    expect(workspaceText(source)).toContain("B Unknown USD");
    expect(() => parseWorkspace(source.replace("b:null", "c:null"))).toThrow();
  });
});

describe("time and dependency correctness", () => {
  it("rejects nonexistent times and requires an explicit repeated-time choice", () => {
    expect(localInstants("2026-03-08T02:30", "America/Los_Angeles")).toEqual(
      [],
    );
    const value = {
      local: "2026-11-01T01:30",
      timeZone: "America/Los_Angeles",
      occurrence: "" as const,
    };
    expect(resolveDateTime(value).error).toContain("twice");
    expect(resolveDateTime({ ...value, occurrence: "earlier" }).instant).toBe(
      "2026-11-01T08:30:00.000Z",
    );
    expect(resolveDateTime({ ...value, occurrence: "later" }).instant).toBe(
      "2026-11-01T09:30:00.000Z",
    );
    expect(localInstants("2026-10-17T09:00", "Asia/Kathmandu")).toEqual([
      "2026-10-17T03:15:00.000Z",
    ]);
  });
  it("links visits, transfers, dates and the exported calendar", () => {
    expect(parseWorkspace(tripSource)).toBeDefined();
    const draft = { ...tripProps, selected: "bridge" };
    const plan = planItinerary(tripProps, draft);
    expect(plan.result?.required).toBe(195);
    const calendar = itineraryCalendar(
      tripProps,
      draft,
      new Date("2026-10-11T12:00:00Z"),
    );
    expect(calendar.match(/BEGIN:VEVENT/g)).toHaveLength(3);
    expect(calendar).toContain("DTSTART:20261017T160000Z");
    expect(calendar).toContain("STATUS:TENTATIVE");
    expect(
      planItinerary(tripProps, { ...draft, tasks: [...draft.tasks].reverse() })
        .error,
    ).toContain("new travel estimate");
    expect(
      planItinerary(tripProps, {
        ...draft,
        days: [{ ...draft.days[0], date: "2026-10-18" }],
      }).error,
    ).toContain("different date");
  });
  it("detects cycles and unfinished prerequisites without treating local progress as execution", () => {
    const tasks: BoardTask[] = [
      {
        id: "a",
        title: "Prepare",
        owner: "",
        due: "",
        status: "todo",
        dependsOn: [],
      },
      {
        id: "b",
        title: "Ship",
        owner: "",
        due: "",
        status: "todo",
        dependsOn: ["a"],
      },
    ];
    expect(taskErrors(tasks)).toEqual([]);
    expect(taskErrors([{ ...tasks[0], dependsOn: ["b"] }, tasks[1]])).toContain(
      "Dependencies contain a cycle.",
    );
    expect(
      taskErrors([tasks[0], { ...tasks[1], status: "doing" }]).join(),
    ).toContain("not complete");
  });
});

describe("multi-day itinerary exports", () => {
  it("uses the supplied transfer for the date actually scheduled", () => {
    const props = {
      ...tripProps,
      days: [
        {
          label: "Day one",
          date: "2026-10-17",
          windows: [{ start: "09:00", end: "10:00" }],
        },
        {
          label: "Day two",
          date: "2026-10-18",
          windows: [{ start: "09:00", end: "17:00" }],
        },
      ],
      legs: [
        ...tripProps.legs,
        { ...tripProps.legs[0], validOn: "2026-10-18", minutes: 30 },
      ],
    };
    const draft = {
      days: props.days,
      tasks: props.tasks,
      timeZone: props.timeZone,
      selected: "bridge",
    };
    const plan = planItinerary(props, draft);
    expect(plan.error).toBe("");
    expect(plan.result?.required).toBe(180);
    expect(
      itineraryCalendar(
        props,
        draft,
        new Date("2026-10-11T12:00:00Z"),
        "session-one:view",
      ),
    ).toContain("UID:session-one:view:");
    expect(itineraryCalendar(props, draft)).toContain(
      "DTSTART:20261018T160000Z",
    );
  });
});

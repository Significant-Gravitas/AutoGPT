import {
  computerState,
  expertSchedules,
  nextRunText,
  runTone,
} from "../helpers";

const BOX = { sandbox_id: "box", started_at: "2026-11-01T09:00:00Z" };

describe("computerState", () => {
  it("leaves the section out when computers are off on this deployment", () => {
    expect(computerState({ e2b_active: false, box: null, screen_on: false })).toBeNull();
  });

  it("names each box state", () => {
    expect(computerState({ e2b_active: true, box: null, screen_on: false })).toBe(
      "Not started",
    );
    expect(
      computerState({
        e2b_active: true,
        box: { ...BOX, state: "paused" },
        screen_on: false,
      }),
    ).toBe("Asleep");
    expect(
      computerState({
        e2b_active: true,
        box: { ...BOX, state: "running" },
        screen_on: false,
      }),
    ).toBe("Awake");
    expect(
      computerState({
        e2b_active: true,
        box: { ...BOX, state: "running" },
        screen_on: true,
      }),
    ).toBe("Awake, screen on");
  });
});

describe("runTone", () => {
  it("maps run statuses to the three marks", () => {
    expect(runTone("completed")).toBe("done");
    expect(runTone("failed")).toBe("failed");
    expect(runTone("running")).toBe("running");
    expect(runTone("queued")).toBe("running");
    expect(runTone("terminated")).toBe("other");
    expect(runTone("review")).toBe("other");
  });
});

describe("nextRunText", () => {
  const now = new Date(2026, 10, 11, 8, 0);

  it("says Paused when the job has no next run", () => {
    expect(nextRunText("", now)).toBe("Paused");
    expect(nextRunText("not a date", now)).toBe("Paused");
  });

  it("says today for a run later today", () => {
    expect(nextRunText(new Date(2026, 10, 11, 9, 0).toISOString(), now)).toMatch(
      /^Next: today /,
    );
  });

  it("uses a weekday within the week and a date beyond it", () => {
    const soon = nextRunText(new Date(2026, 10, 13, 9, 0).toISOString(), now);
    expect(soon).toMatch(/^Next: \S+ /);
    expect(soon).not.toMatch(/today/);
    const later = nextRunText(new Date(2026, 10, 30, 9, 0).toISOString(), now);
    expect(later).toMatch(/^Next: 30 /);
  });
});

describe("expertSchedules", () => {
  const at = (hour: number) => new Date(2026, 10, 12, hour).toISOString();

  it("keeps this expert's schedules, soonest first and paused last", () => {
    const result = expertSchedules({
      expertID: "max",
      detail: {
        id: "max",
        schedules_paused_at: null,
        workflows: [{ schedule_id: "installed" }, { schedule_id: null }],
      },
      graphSchedules: [
        { id: "paused", name: "Paused", next_run_time: "", expert_id: "max" },
        { id: "late", name: "Late", next_run_time: at(18), expert_id: "max" },
        { id: "installed", name: "Installed", next_run_time: at(12) },
        { id: "other", name: "Other", next_run_time: at(7), expert_id: "mina" },
      ],
      followupSchedules: [
        { id: "chat", name: "Chat", next_run_time: at(9), expert_id: "max" },
        { id: "otto", name: "Otto", next_run_time: at(8), expert_id: null },
      ],
    });
    expect(result.map((schedule) => schedule.id)).toEqual([
      "chat",
      "installed",
      "late",
      "paused",
    ]);
  });

  it("still matches stamped schedules before the expert loads", () => {
    const result = expertSchedules({
      expertID: "max",
      detail: undefined,
      graphSchedules: [{ id: "a", name: "A", next_run_time: at(9), expert_id: "max" }],
      followupSchedules: [],
    });
    expect(result).toHaveLength(1);
  });
});

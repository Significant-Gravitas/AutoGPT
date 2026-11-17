import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import type { SuggestedTaskState } from "@/app/api/__generated__/models/suggestedTaskState";
import { beforeEach, describe, expect, it } from "vitest";
import type { ExpertIdentity } from "../../../useExpertMap";
import {
  getMentionedProviders,
  getStartHereID,
  getTaskOwner,
  readCollapsed,
  splitTasks,
  writeCollapsed,
  withState,
} from "../helpers";

const COLLAPSED_KEY = "home.suggestedTasks.collapsed";

const MARIA: ExpertIdentity = {
  id: "expert-maria",
  name: "Maria",
  avatarUrl: null,
  role: "Finance, Invoicing & Bookkeeping",
  isArchived: false,
  readOnlyReason: null,
};

const SAM: ExpertIdentity = {
  id: "expert-sam",
  name: "Sam",
  avatarUrl: null,
  role: "Sales",
  isArchived: false,
  readOnlyReason: null,
};

function task(
  id: string,
  state?: SuggestedTaskState,
  extra: Partial<SuggestedTask> = {},
): SuggestedTask {
  return {
    id,
    title: `Task ${id}`,
    prompt: `Do task ${id}.`,
    ...(state ? { state } : {}),
    ...extra,
  };
}

function ids(tasks: SuggestedTask[]) {
  return tasks.map((item) => item.id);
}

describe("getMentionedProviders", () => {
  it.each([
    ["No tools named here", []],
    ["Plan Monday's work with teams on a medium budget", []],
    ["Build a linear model and mark X on the chart", []],
    [
      "Summarize GITHUB updates in Slack, then GitHub again",
      ["github", "slack"],
    ],
    ["Sync Microsoft Teams with Monday.com", ["teams", "monday"]],
    ["Read Gmail and Google Calendar", ["google"]],
    ["Find Jira, Confluence, and Atlassian issues", ["atlassian"]],
    [
      "Send Slack, GitHub, Linear, and Notion updates",
      ["slack", "github", "linear"],
    ],
    ["Review githubish and slackbot tools", []],
  ])("finds providers in %s", (title, expected) => {
    expect(getMentionedProviders(title)).toEqual(expected);
  });
});

describe("splitTasks", () => {
  it("shows the first three tasks and keeps the rest for later while none has been touched", () => {
    const { top, more, hasActedOnTop } = splitTasks([
      task("a", "open"),
      task("b", "open"),
      task("c", "open"),
      task("d", "open"),
      task("e", "open"),
    ]);

    expect(ids(top)).toEqual(["a", "b", "c"]);
    expect(ids(more)).toEqual(["d", "e"]);
    expect(hasActedOnTop).toBe(false);
  });

  it("keeps a removed task's place in the first three instead of pulling a later one up", () => {
    const { top, more } = splitTasks([
      task("a", "open"),
      task("b", "dismissed"),
      task("c", "open"),
      task("d", "open"),
    ]);

    expect(ids(top)).toEqual(["a", "c"]);
    expect(ids(more)).toEqual(["d"]);
  });

  it("leaves removed tasks out of the later ones but keeps the started ones", () => {
    const { more } = splitTasks([
      task("a", "open"),
      task("b", "open"),
      task("c", "open"),
      task("d", "dismissed"),
      task("e", "sent", { session_id: "chat-e" }),
      task("f", "open"),
    ]);

    expect(ids(more)).toEqual(["e", "f"]);
  });

  it("counts a started task in the first three as acted on", () => {
    const { top, hasActedOnTop } = splitTasks([
      task("a", "sent", { session_id: "chat-a" }),
      task("b", "open"),
      task("c", "open"),
      task("d", "open"),
    ]);

    expect(ids(top)).toEqual(["a", "b", "c"]);
    expect(hasActedOnTop).toBe(true);
  });

  it("counts a removed task in the first three as acted on", () => {
    expect(
      splitTasks([task("a", "open"), task("b", "open"), task("c", "dismissed")])
        .hasActedOnTop,
    ).toBe(true);
  });

  it("does not count a started or removed later task as acting on the first three", () => {
    const { hasActedOnTop } = splitTasks([
      task("a", "open"),
      task("b", "open"),
      task("c", "open"),
      task("d", "sent", { session_id: "chat-d" }),
      task("e", "dismissed"),
    ]);

    expect(hasActedOnTop).toBe(false);
  });

  it("reads a task stored without a state as open", () => {
    const { top, more, hasActedOnTop } = splitTasks([
      task("a"),
      task("b"),
      task("c"),
      task("d"),
    ]);

    expect(ids(top)).toEqual(["a", "b", "c"]);
    expect(ids(more)).toEqual(["d"]);
    expect(hasActedOnTop).toBe(false);
  });

  it("has nothing for later when there are three tasks or fewer", () => {
    expect(splitTasks([task("a", "open"), task("b", "open")])).toEqual({
      top: [task("a", "open"), task("b", "open")],
      more: [],
      hasActedOnTop: false,
    });
    expect(splitTasks([])).toEqual({
      top: [],
      more: [],
      hasActedOnTop: false,
    });
  });
});

describe("getStartHereID", () => {
  it("starts on the first open task in the order shown", () => {
    expect(
      getStartHereID([
        task("a", "sent", { session_id: "chat-a" }),
        task("b", "open"),
        task("c", "open"),
      ]),
    ).toBe("b");
  });

  it("follows the order it is given, not the ids", () => {
    expect(getStartHereID([task("z", "open"), task("a", "open")])).toBe("z");
  });

  it("skips removed tasks", () => {
    expect(getStartHereID([task("a", "dismissed"), task("b", "open")])).toBe(
      "b",
    );
  });

  it("starts on a task stored without a state", () => {
    expect(
      getStartHereID([task("a", "sent", { session_id: "chat-a" }), task("b")]),
    ).toBe("b");
  });

  it("has no starting point once every task is started or removed", () => {
    expect(
      getStartHereID([
        task("a", "sent", { session_id: "chat-a" }),
        task("b", "dismissed"),
      ]),
    ).toBeNull();
    expect(getStartHereID([])).toBeNull();
  });
});

describe("getTaskOwner", () => {
  it("gives a task to the hired expert it names", () => {
    expect(
      getTaskOwner(task("a", "open", { expert_id: SAM.id }), [MARIA, SAM]),
    ).toEqual({ kind: "expert", expert: SAM });
  });

  it("gives a task with no owner to Otto", () => {
    expect(getTaskOwner(task("a", "open"), [MARIA, SAM])).toEqual({
      kind: "otto",
    });
    expect(
      getTaskOwner(task("a", "open", { expert_id: null }), [MARIA]),
    ).toEqual({ kind: "otto" });
  });

  it("gives a task to Otto when its expert is no longer on the team", () => {
    expect(
      getTaskOwner(
        task("a", "open", {
          expert_id: "expert-fired",
          expert_template_id: "template-bookkeeper",
          expert_name: "Priya",
        }),
        [MARIA, SAM],
      ),
    ).toEqual({ kind: "otto" });
  });

  it("gives a task to Otto when the team is empty", () => {
    expect(
      getTaskOwner(task("a", "open", { expert_id: MARIA.id }), []),
    ).toEqual({ kind: "otto" });
  });

  it("keeps a started task with the expert its chat is with, fired since or not", () => {
    const fired = { ...SAM, isArchived: true };
    const everyExpert = new Map([
      [MARIA.id, MARIA],
      [fired.id, fired],
    ]);
    const started = task("a", "sent", {
      expert_id: SAM.id,
      session_id: "session-sam",
    });

    expect(getTaskOwner(started, [MARIA], everyExpert)).toEqual({
      kind: "expert",
      expert: fired,
    });
    expect(
      getTaskOwner(
        { ...started, state: "open", session_id: null },
        [MARIA],
        everyExpert,
      ),
    ).toEqual({ kind: "otto" });
  });
});

describe("readCollapsed and writeCollapsed", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("starts open for a user who never collapsed the list", () => {
    expect(readCollapsed("user-1")).toBe(false);
  });

  it("remembers a collapsed list for the user who collapsed it", () => {
    writeCollapsed("user-1", true);

    expect(readCollapsed("user-1")).toBe(true);
    expect(readCollapsed("user-2")).toBe(false);
    expect(
      JSON.parse(window.localStorage.getItem(COLLAPSED_KEY) ?? ""),
    ).toEqual(["user-1"]);
  });

  it("opens the list again when the user expands it", () => {
    writeCollapsed("user-1", true);
    writeCollapsed("user-1", false);

    expect(readCollapsed("user-1")).toBe(false);
    expect(
      JSON.parse(window.localStorage.getItem(COLLAPSED_KEY) ?? ""),
    ).toEqual([]);
  });

  it("keeps one user's choice when another user on the same browser changes theirs", () => {
    writeCollapsed("user-1", true);
    writeCollapsed("user-2", true);
    writeCollapsed("user-2", false);
    writeCollapsed("user-3", true);

    expect(readCollapsed("user-1")).toBe(true);
    expect(readCollapsed("user-2")).toBe(false);
    expect(readCollapsed("user-3")).toBe(true);
  });

  it("does not list a user twice when they collapse the list again", () => {
    writeCollapsed("user-1", true);
    writeCollapsed("user-1", true);

    expect(
      JSON.parse(window.localStorage.getItem(COLLAPSED_KEY) ?? ""),
    ).toEqual(["user-1"]);
  });

  it.each([
    ["text that is not JSON", "{not json"],
    ["an object", JSON.stringify({ "user-1": true })],
    ["a string", JSON.stringify("user-1")],
    ["null", "null"],
  ])("reads %s in storage as not collapsed", (_, value) => {
    window.localStorage.setItem(COLLAPSED_KEY, value);

    expect(readCollapsed("user-1")).toBe(false);
  });

  it("ignores entries that are not user ids", () => {
    window.localStorage.setItem(
      COLLAPSED_KEY,
      JSON.stringify([1, null, { id: "user-2" }, "user-1"]),
    );

    expect(readCollapsed("user-1")).toBe(true);
    expect(readCollapsed("user-2")).toBe(false);
  });

  it("recovers from malformed storage on the next write", () => {
    window.localStorage.setItem(COLLAPSED_KEY, "{not json");

    writeCollapsed("user-1", true);

    expect(readCollapsed("user-1")).toBe(true);
    expect(
      JSON.parse(window.localStorage.getItem(COLLAPSED_KEY) ?? ""),
    ).toEqual(["user-1"]);
  });

  it("never collapses the list for a signed-out visitor and stores nothing for them", () => {
    writeCollapsed(null, true);

    expect(readCollapsed(null)).toBe(false);
    expect(window.localStorage.getItem(COLLAPSED_KEY)).toBeNull();
  });

  it("leaves other users' choices alone when there is no user", () => {
    writeCollapsed("user-1", true);
    writeCollapsed(null, false);

    expect(readCollapsed("user-1")).toBe(true);
    expect(readCollapsed(null)).toBe(false);
  });
});

describe("withState", () => {
  it("marks a sent task with its chat and who took it", () => {
    const tasks = [
      task("a", "open", { expert_id: "expert-fired" }),
      task("b", "open"),
    ];

    expect(
      withState({
        tasks,
        ids: ["a"],
        state: "sent",
        sent: { sessionID: "session-otto", expertID: null },
      }),
    ).toEqual([
      {
        ...tasks[0],
        state: "sent",
        session_id: "session-otto",
        expert_id: null,
      },
      tasks[1],
    ]);
    expect(withState({ tasks, ids: ["b"], state: "dismissed" })).toEqual([
      tasks[0],
      { ...tasks[1], state: "dismissed" },
    ]);
  });
});

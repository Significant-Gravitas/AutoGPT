import type { PlanResponse } from "@/app/api/__generated__/models/planResponse";
import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  consentLine,
  getStepTasks,
  getTaskOwner,
  isCheckoutReturn,
  joinNames,
  readShownPlan,
  rememberShownPlan,
  roleHeard,
} from "../helpers";

const SHARPEN_BUSINESS =
  "Tell me about your business once you're in and I'll sharpen it.";
const SHARPEN_WORK =
  "Tell me about your work once you're in and I'll sharpen it.";
const NOTHING_RUNS = "Nothing runs until you pick a task.";

function expert(name: string, templateID: string): RecommendedExpert {
  return { template_id: templateID, name, role: "Sales", avatar_url: null };
}

function task(id: string, ownerID: string | null = null): SuggestedTask {
  return {
    id,
    title: `Task ${id}`,
    prompt: `Do task ${id}`,
    expert_template_id: ownerID,
  };
}

const PLAN: PlanResponse = {
  ready: true,
  source: "llm",
  skipped_dump: false,
  heard: "You run a small bakery and spend your evenings on invoices.",
  team: [expert("Maria", "tpl-maria")],
  tasks: [task("t1", "tpl-maria"), task("t2")],
};

describe("roleHeard", () => {
  test.each([
    ["Founder/CEO", "Founder / CEO", "a founder"],
    ["Operations", "Operations", "an operations lead"],
    ["Sales/BD", "Sales / BD", "a sales lead"],
    ["Marketing", "Marketing", "a marketer"],
    ["Product/PM", "Product / PM", "a product manager"],
    ["Engineering", "Engineering", "an engineer"],
    ["HR/People", "HR / People", "a people lead"],
  ])("names a skipped %s dump's role and noun", (role, label, noun) => {
    expect(roleHeard({ role, skippedDump: true })).toBe(
      `${label}. You skipped the brain dump, so this is a first plan for ${noun}. ${SHARPEN_BUSINESS}`,
    );
  });

  test.each(["Other", "Astronaut", ""])(
    "falls back to a role-free line for the %j role",
    (role) => {
      expect(roleHeard({ role, skippedDump: true })).toBe(
        `You skipped the brain dump, so this is a first plan to get going. ${SHARPEN_WORK}`,
      );
    },
  );

  test("drops the skip clause when the dump was not skipped", () => {
    expect(roleHeard({ role: "Marketing", skippedDump: false })).toBe(
      `Marketing. This is a first plan for a marketer. ${SHARPEN_BUSINESS}`,
    );
    expect(roleHeard({ role: "Engineering", skippedDump: false })).toBe(
      `Engineering. This is a first plan for an engineer. ${SHARPEN_BUSINESS}`,
    );
    expect(roleHeard({ role: "Other", skippedDump: false })).toBe(
      `This is a first plan to get going. ${SHARPEN_WORK}`,
    );
  });
});

describe("joinNames", () => {
  test("joins names without an Oxford comma", () => {
    expect(joinNames([])).toBe("");
    expect(joinNames(["Maria"])).toBe("Maria");
    expect(joinNames(["Maria", "Max"])).toBe("Maria and Max");
    expect(joinNames(["Maria", "Max", "Mina"])).toBe("Maria, Max and Mina");
    expect(joinNames(["Maria", "Max", "Mina", "Leo"])).toBe(
      "Maria, Max, Mina and Leo",
    );
  });
});

describe("consentLine", () => {
  test("names every expert who joins when the trial starts", () => {
    expect(consentLine(["Maria", "Max", "Mina"], "trial")).toBe(
      `Maria, Max and Mina join your team when your trial starts. ${NOTHING_RUNS}`,
    );
  });

  test("uses the singular verb for one expert", () => {
    expect(consentLine(["Maria"], "trial")).toBe(
      `Maria joins your team when your trial starts. ${NOTHING_RUNS}`,
    );
  });

  test("names the moment the team joins", () => {
    expect(consentLine(["Maria", "Max"], "subscribe")).toBe(
      `Maria and Max join your team when you subscribe. ${NOTHING_RUNS}`,
    );
    expect(consentLine(["Maria"], "continue")).toBe(
      `Maria joins your team when you continue. ${NOTHING_RUNS}`,
    );
  });

  test("leaves the work with Otto when nobody joins, whatever the moment", () => {
    const line = `Otto takes these on for now. You can hire experts any time. ${NOTHING_RUNS}`;
    expect(consentLine([], "trial")).toBe(line);
    expect(consentLine([], "subscribe")).toBe(line);
    expect(consentLine([], "continue")).toBe(line);
  });
});

describe("getTaskOwner", () => {
  const team = [expert("Maria", "tpl-maria"), expert("Max", "tpl-max")];

  test("finds the team member the task names", () => {
    expect(getTaskOwner(task("t1", "tpl-max"), team)).toBe(team[1]);
  });

  test("gives Otto a task with no owner", () => {
    expect(getTaskOwner(task("t1", null), team)).toBeNull();
    expect(
      getTaskOwner({ id: "t2", title: "Untouched", prompt: "Go" }, team),
    ).toBeNull();
  });

  test("gives Otto a task whose owner is not on the team", () => {
    expect(getTaskOwner(task("t1", "tpl-mina"), team)).toBeNull();
    expect(getTaskOwner(task("t1", "tpl-maria"), [])).toBeNull();
  });
});

describe("getStepTasks", () => {
  test("takes the first three tasks, in order", () => {
    const tasks = ["a", "b", "c", "d", "e"].map((id) => task(id));
    expect(getStepTasks({ ...PLAN, tasks }).map((t) => t.id)).toEqual([
      "a",
      "b",
      "c",
    ]);
  });

  test("takes fewer when the plan has fewer, and none when it has none", () => {
    expect(getStepTasks({ ...PLAN, tasks: [task("a")] })).toHaveLength(1);
    expect(getStepTasks({ ...PLAN, tasks: undefined })).toEqual([]);
  });
});

describe("isCheckoutReturn", () => {
  test("recognises a cancelled subscription or trial checkout", () => {
    expect(
      isCheckoutReturn(new URLSearchParams("subscription=cancelled")),
    ).toBe(true);
    expect(isCheckoutReturn(new URLSearchParams("trial=cancelled"))).toBe(true);
  });

  test("ignores any other visit", () => {
    expect(isCheckoutReturn(new URLSearchParams())).toBe(false);
    expect(isCheckoutReturn(new URLSearchParams("subscription=success"))).toBe(
      false,
    );
    expect(isCheckoutReturn(new URLSearchParams("trial=started"))).toBe(false);
  });
});

describe("remembering the shown plan", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  test("stores the plan per user and reads it back", () => {
    rememberShownPlan("user-1", PLAN);

    expect(
      JSON.parse(sessionStorage.getItem("autogpt:onboarding-plan:user-1")!),
    ).toEqual(PLAN);
    expect(readShownPlan("user-1")).toEqual(PLAN);
  });

  test("never shows one user's plan to another", () => {
    rememberShownPlan("user-1", PLAN);

    expect(readShownPlan("user-2")).toBeNull();
  });

  test("does nothing without a user", () => {
    rememberShownPlan(null, PLAN);

    expect(sessionStorage.length).toBe(0);
    expect(readShownPlan(null)).toBeNull();
  });

  test("reads nothing when nothing was stored", () => {
    expect(readShownPlan("user-1")).toBeNull();
  });

  test("ignores a stored value that is not a plan", () => {
    const key = "autogpt:onboarding-plan:user-1";
    for (const value of [
      "{not json",
      "null",
      '"a string"',
      JSON.stringify({ ...PLAN, tasks: undefined }),
      JSON.stringify({ ...PLAN, team: "Maria" }),
    ]) {
      sessionStorage.setItem(key, value);
      expect(readShownPlan("user-1")).toBeNull();
    }
  });

  test("survives a browser that blocks session storage", () => {
    function blocked(): never {
      throw new DOMException("blocked", "SecurityError");
    }
    vi.stubGlobal("sessionStorage", { getItem: blocked, setItem: blocked });

    expect(() => rememberShownPlan("user-1", PLAN)).not.toThrow();
    expect(readShownPlan("user-1")).toBeNull();
  });
});

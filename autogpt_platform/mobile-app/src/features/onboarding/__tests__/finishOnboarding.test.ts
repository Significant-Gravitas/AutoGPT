import { ApiError } from "@/lib/api/client";
import type { BrainDumpBody } from "@/lib/api/onboarding";

import {
  PLAN_POLL_MS,
  PLAN_WAIT_MS,
  checklistFor,
  finishOnboarding,
  joinNames,
  type OnboardingAPI,
} from "../finishOnboarding";
import type { WorkAnswer } from "../helpers";

function fakeAPI(overrides: Partial<OnboardingAPI> = {}) {
  const calls: string[] = [];
  const dumps: BrainDumpBody[] = [];
  const api: OnboardingAPI = {
    submitProfile: async () => {
      calls.push("profile");
    },
    finalizeDump: async (body) => {
      calls.push("dump");
      dumps.push(body);
    },
    isPaywalled: async () => {
      calls.push("paywall");
      return false;
    },
    getPlan: async () => {
      calls.push("plan");
      return { ready: true, team: [{ name: "Maria" }, { name: "Max" }] };
    },
    hirePlan: async () => {
      calls.push("hire");
      return { failed: [] };
    },
    complete: async () => {
      calls.push("complete");
    },
    ...overrides,
  };
  return { api, calls, dumps };
}

const base = {
  userName: "Ada",
  role: "Founder/CEO",
  recordingID: "rec-1",
  onStage: () => undefined,
  wait: async () => undefined,
};
const typed: WorkAnswer = { mode: "typed", text: "I chase invoices." };

describe("finishOnboarding", () => {
  it("saves the answers, hires the team and finishes, in that order", async () => {
    const { api, calls, dumps } = fakeAPI();
    const result = await finishOnboarding({ ...base, api, work: typed });
    expect(calls).toEqual(["profile", "dump", "paywall", "plan", "hire", "complete"]);
    expect(dumps[0]).toEqual({
      recording_id: "rec-1",
      input_mode: "typed",
      text: "I chase invoices.",
    });
    expect(result).toEqual({ kind: "done", notHired: [] });
  });

  it("records a skip as a skip", async () => {
    const { api, dumps } = fakeAPI();
    await finishOnboarding({ ...base, api, work: { mode: "skipped" } });
    expect(dumps[0]).toEqual({ recording_id: "rec-1", input_mode: "skipped" });
  });

  it("stops before hiring on a server that wants a plan first", async () => {
    const { api, calls } = fakeAPI({ isPaywalled: async () => true });
    const result = await finishOnboarding({ ...base, api, work: typed });
    expect(result).toEqual({ kind: "planNeeded" });
    expect(calls).not.toContain("hire");
    expect(calls).not.toContain("complete");
  });

  it("treats a 402 on the hire the same way", async () => {
    const { api, calls } = fakeAPI({
      hirePlan: async () => {
        throw new ApiError(402, "Plan required");
      },
    });
    expect(await finishOnboarding({ ...base, api, work: typed })).toEqual({
      kind: "planNeeded",
    });
    expect(calls).not.toContain("complete");
  });

  it("finishes without the team when the hire fails", async () => {
    const { api } = fakeAPI({
      hirePlan: async () => {
        throw new ApiError(500, "down");
      },
    });
    expect(await finishOnboarding({ ...base, api, work: typed })).toEqual({
      kind: "done",
      notHired: ["Maria", "Max"],
    });
  });

  it("reports who could not join", async () => {
    const { api } = fakeAPI({ hirePlan: async () => ({ failed: [{ name: "Max" }] }) });
    expect(await finishOnboarding({ ...base, api, work: typed })).toEqual({
      kind: "done",
      notHired: ["Max"],
    });
  });

  it("stops waiting for a plan that never comes", async () => {
    let asked = 0;
    const { api, calls } = fakeAPI({
      getPlan: async () => {
        asked += 1;
        return { ready: false, team: [{ name: "Maria" }] };
      },
    });
    await finishOnboarding({ ...base, api, work: typed });
    expect(asked).toBeLessThanOrEqual(PLAN_WAIT_MS / PLAN_POLL_MS + 1);
    expect(calls).toContain("hire");
  });

  it("fails, to be tried again, when an answer can't be saved", async () => {
    const { api, calls } = fakeAPI({
      finalizeDump: async () => {
        throw new ApiError(500, "down");
      },
    });
    await expect(finishOnboarding({ ...base, api, work: typed })).rejects.toThrow();
    expect(calls).not.toContain("complete");
  });

  it("sends the same recording id on a second try", async () => {
    const { api, dumps } = fakeAPI();
    await finishOnboarding({ ...base, api, work: typed });
    await finishOnboarding({ ...base, api, work: typed });
    expect(dumps.map((dump) => dump.recording_id)).toEqual(["rec-1", "rec-1"]);
  });
});

describe("joinNames", () => {
  it("reads as a sentence", () => {
    expect(joinNames(["Maria"])).toBe("Maria");
    expect(joinNames(["Maria", "Max"])).toBe("Maria and Max");
    expect(joinNames(["Maria", "Max", "Riley"])).toBe("Maria, Max and Riley");
  });
});

describe("checklistFor", () => {
  it("says what is really happening", () => {
    expect(checklistFor(typed, ["Maria", "Max"])).toEqual([
      "Reading your brain dump",
      "Hiring Maria and Max",
      "Building your space",
    ]);
    expect(checklistFor({ mode: "skipped" }, [])).toEqual([
      "Personalizing your experience",
      "Setting up your team",
      "Building your space",
    ]);
  });
});

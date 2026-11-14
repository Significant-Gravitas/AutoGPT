import { FRONTEND_URL } from "@/lib/env";

import {
  firstName,
  greetingLine,
  hasExistingWork,
  homeLinkTarget,
  inputPlaceholder,
  introParts,
  nextRunLabel,
  ottoWording,
  sentenceWords,
} from "../helpers";

const SESSION = "0b6f3c1e-5d2a-4c8b-9f1e-2a3b4c5d6e7f";

describe("homeLinkTarget", () => {
  it("opens a chat for a copilot link with a session id", () => {
    expect(homeLinkTarget(`/copilot?sessionId=${SESSION}`)).toEqual({
      kind: "chat",
      sessionID: SESSION,
    });
  });

  it("opens the web page when the session id is not a UUID", () => {
    expect(homeLinkTarget("/copilot?sessionId=not-a-uuid")).toEqual({
      kind: "web",
      url: `${FRONTEND_URL}/copilot?sessionId=not-a-uuid`,
    });
  });

  it("opens the team screen for /team", () => {
    expect(homeLinkTarget("/team")).toEqual({ kind: "team" });
  });

  it("opens an expert's page on the web", () => {
    expect(homeLinkTarget("/team/abc")).toEqual({
      kind: "web",
      url: `${FRONTEND_URL}/team/abc`,
    });
  });

  it("opens other same-site paths on the web", () => {
    expect(homeLinkTarget("/profile/credits")).toEqual({
      kind: "web",
      url: `${FRONTEND_URL}/profile/credits`,
    });
  });

  it("refuses a scheme-relative link", () => {
    expect(homeLinkTarget("//evil.example.com")).toBeNull();
  });

  it("refuses a foreign link", () => {
    expect(homeLinkTarget("https://evil.example.com")).toBeNull();
  });

  it("refuses an empty link", () => {
    expect(homeLinkTarget("")).toBeNull();
  });
});

describe("firstName", () => {
  it("takes the first word of the name", () => {
    expect(firstName("  Sam Lee ", "sam@example.com")).toBe("Sam");
  });

  it("ignores a name that is only the email's local part", () => {
    expect(firstName("sam.lee", "Sam.Lee@example.com")).toBeNull();
  });

  it("has nothing for a missing or blank name", () => {
    expect(firstName(null)).toBeNull();
    expect(firstName("   ")).toBeNull();
  });
});

describe("greetingLine", () => {
  it("greets by name", () => {
    expect(greetingLine("Sam")).toBe("Hey, Sam");
  });

  it("falls back without a name", () => {
    expect(greetingLine(null)).toBe("Hey there");
  });
});

describe("introParts", () => {
  it("asks about the work for Otto", () => {
    expect(introParts(null)).toEqual({
      before: "Tell",
      after: " about your work, and it will find what to automate.",
    });
  });

  it("uses a role that names a person as written", () => {
    expect(introParts({ name: "Max", role: "Marketer" })).toEqual({
      before: "I'm",
      after: ", your Marketer. What should I take on?",
    });
  });

  it("adds expert to a domain role", () => {
    expect(introParts({ name: "Max", role: "Support" }).after).toBe(
      ", your Support expert. What should I take on?",
    );
    expect(introParts({ name: "Max", role: "Customer Success" }).after).toBe(
      ", your Customer Success expert. What should I take on?",
    );
  });

  it("maps a preset role to its display label", () => {
    expect(introParts({ name: "Max", role: "Sales" }).after).toBe(
      ", your Sales Development Rep. What should I take on?",
    );
  });

  it("leaves the role out when there is none", () => {
    expect(introParts({ name: "Max", role: null }).after).toBe(
      ". What should I take on?",
    );
    expect(introParts({ name: "Max", role: "  " }).after).toBe(
      ". What should I take on?",
    );
  });
});

describe("sentenceWords", () => {
  it("keeps a leading mark apart from the words", () => {
    expect(sentenceWords(", your Marketer. What")).toEqual({
      mark: ",",
      words: ["your", "Marketer.", "What"],
    });
  });

  it("has no mark for a part that starts with a space", () => {
    expect(sentenceWords(" about your work.")).toEqual({
      mark: "",
      words: ["about", "your", "work."],
    });
  });
});

describe("inputPlaceholder", () => {
  it("uses the web's narrow placeholder for Otto", () => {
    expect(inputPlaceholder(null)).toBe("I'm a chef and I hate...");
  });

  it("names the expert", () => {
    expect(inputPlaceholder({ name: "Max" })).toBe("What should Max work on?");
    expect(inputPlaceholder({ name: "Max" }, true)).toBe("What should Max work on?");
  });

  it("asks about the team once the account has work going", () => {
    expect(inputPlaceholder(null, true)).toBe(
      "What should the team work on? e.g. 'Follow up with yesterday's leads'",
    );
  });
});

describe("hasExistingWork", () => {
  const empty = {
    attention: [],
    briefing: { completed_count: 0 },
    active_tasks: [],
    upcoming_tasks: [],
    team: { total: 0 },
    week: { run_count: 0 },
  };

  it("is false with no dashboard and for an untouched account", () => {
    expect(hasExistingWork(undefined)).toBe(false);
    expect(hasExistingWork(empty)).toBe(false);
    expect(hasExistingWork({ ...empty, recent_work: null })).toBe(false);
  });

  it.each([
    ["a hired expert", { team: { total: 1 } }],
    ["a run this week", { week: { run_count: 3 } }],
    ["recent work", { recent_work: { total_count: 2 } }],
    ["a recent work group", { recent_work: { groups: [{}] } }],
  ])("is true with %s", (_label, patch) => {
    expect(hasExistingWork({ ...empty, ...patch })).toBe(true);
  });
});

describe("nextRunLabel", () => {
  // Wednesday 11 November 2026, 10:00 local time.
  const now = new Date(2026, 10, 11, 10, 0);

  function at(day: number, hour: number) {
    return new Date(2026, 10, day, hour, 0).toISOString();
  }

  it("says Today for later the same day", () => {
    expect(nextRunLabel(at(11, 16), now, "en-GB")).toBe("Today 16:00");
  });

  it("says Tomorrow for the next day", () => {
    expect(nextRunLabel(at(12, 9), now, "en-GB")).toBe("Tomorrow 9:00");
  });

  it("names the weekday within a week", () => {
    expect(nextRunLabel(at(16, 9), now, "en-GB")).toBe("Monday 9:00");
  });

  it("gives the date a week or more out", () => {
    expect(nextRunLabel(at(20, 9), now, "en-GB")).toBe("20 Nov 9:00");
  });
});

describe("ottoWording", () => {
  it("names Otto instead of the product", () => {
    expect(ottoWording("What reporting features can AutoGPT enhance?")).toBe(
      "What reporting features can Otto enhance?",
    );
  });

  it("leaves other words alone", () => {
    expect(ottoWording("Plan my AutoGPTish week")).toBe("Plan my AutoGPTish week");
  });
});

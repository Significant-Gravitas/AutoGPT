import { FRONTEND_URL } from "@/lib/env";

import { briefingLine, greeting, homeLinkTarget, nextRunLabel } from "../helpers";

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

describe("greeting", () => {
  it("says morning before noon with the first name", () => {
    expect(greeting("Sam Lee", new Date(2026, 10, 11, 11, 59))).toBe(
      "Good morning, Sam",
    );
  });

  it("says afternoon from noon", () => {
    expect(greeting("Sam", new Date(2026, 10, 11, 12, 0))).toBe(
      "Good afternoon, Sam",
    );
  });

  it("says evening from 18:00 and leaves out a missing name", () => {
    expect(greeting(null, new Date(2026, 10, 11, 18, 0))).toBe("Good evening");
    expect(greeting("  ", new Date(2026, 10, 11, 18, 0))).toBe("Good evening");
  });
});

describe("briefingLine", () => {
  it("counts one thing", () => {
    expect(briefingLine(1)).toBe("Your team finished 1 thing in the last day.");
  });

  it("counts several things", () => {
    expect(briefingLine(4)).toBe("Your team finished 4 things in the last day.");
  });

  it("falls back when nothing finished or nothing loaded", () => {
    expect(briefingLine(0)).toBe("What should we work on?");
    expect(briefingLine(undefined)).toBe("What should we work on?");
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

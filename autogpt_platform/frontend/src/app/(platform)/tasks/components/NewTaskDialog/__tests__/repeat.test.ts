import { describe, expect, it } from "vitest";
import { cronOf, rhythmOf, type RepeatChoice } from "../repeat";
import { fieldOfRefusal, optionsRequest, refusalText } from "../taskOptions";

function choice(extra: Partial<RepeatChoice>): RepeatChoice {
  return { kind: "daily", time: "09:00", day: 1, ...extra };
}

describe("Repeat rhythms", () => {
  it.each([
    [choice({ kind: "hourly" }), "0 * * * *", "Every hour"],
    [
      choice({ kind: "daily", time: "07:30" }),
      "30 7 * * *",
      "Every day at 07:30",
    ],
    [
      choice({ kind: "weekdays", time: "09:00" }),
      "0 9 * * 1-5",
      "Every weekday at 09:00",
    ],
    [
      choice({ kind: "weekly", time: "17:05", day: 5 }),
      "5 17 * * 5",
      "Every Friday at 17:05",
    ],
  ])("builds the cron and words for %o", (repeat, cron, words) => {
    expect(cronOf(repeat)).toBe(cron);
    expect(rhythmOf(repeat)).toBe(words);
  });

  it("builds no cron from a time that does not read", () => {
    expect(cronOf(choice({ time: "25:00" }))).toBeNull();
    expect(
      optionsRequest({
        startAt: null,
        repeat: choice({ time: "" }),
        doneWhen: "",
        goal: "",
      }),
    ).toEqual({});
  });

  it("puts each refusal by the part it is about", () => {
    expect(fieldOfRefusal("Choose a start time in the future.")).toBe("start");
    expect(fieldOfRefusal("Give a start time or a repeat, not both.")).toBe(
      "start",
    );
    expect(fieldOfRefusal("Each cron has 5 fields: minute hour…")).toBe(
      "repeat",
    );
    expect(fieldOfRefusal("Title is too long.")).toBe("form");
    expect(refusalText("Value error, Choose a start time in the future.")).toBe(
      "Choose a start time in the future.",
    );
  });
});

import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { describe, expect, it } from "vitest";
import {
  atResetRefusal,
  returnToDraft,
  SCHEDULE_FAILED_TITLE,
  scheduledSendLabel,
  sendAtResetFailureTitle,
  SentNowError,
} from "../helpers";

const NOW = new Date(2026, 9, 11, 20, 48); // Sunday 8:48 PM, local

describe("scheduledSendLabel", () => {
  it("gives the time alone for a send later today", () => {
    expect(scheduledSendLabel(new Date(2026, 9, 11, 23, 5), NOW)).toBe(
      "Sends at 11:05 PM",
    );
  });

  it("names the day for a send on another day", () => {
    expect(scheduledSendLabel(new Date(2026, 9, 12, 1, 7), NOW)).toBe(
      "Sends Monday at 1:07 AM",
    );
  });
});

describe("atResetRefusal", () => {
  it.each(["not_out_of_budget", "no_reset"])("reads a %s refusal", (detail) => {
    expect(atResetRefusal(new ApiError("refused", 409, { detail }))).toBe(
      detail,
    );
  });

  it("leaves any other failure alone", () => {
    expect(atResetRefusal(new ApiError("busy", 409, { detail: "busy" }))).toBe(
      null,
    );
    expect(
      atResetRefusal(new ApiError("down", 503, { detail: "no_reset" })),
    ).toBe(null);
    expect(atResetRefusal(new Error("offline"))).toBe(null);
  });
});

describe("returnToDraft", () => {
  it("keeps a draft written meanwhile after the returned message", () => {
    expect(returnToDraft("", "Scheduled")).toBe("Scheduled");
    expect(returnToDraft("Scheduled", "Scheduled")).toBe("Scheduled");
    expect(returnToDraft("New thought", "Scheduled")).toBe(
      "Scheduled\n\nNew thought",
    );
  });
});

describe("sendAtResetFailureTitle", () => {
  it("calls a send that went out now a send", () => {
    const error = new SentNowError(new Error("Network down"));
    expect(sendAtResetFailureTitle(error)).toBe("Couldn't send message");
    expect(error.message).toBe("Network down");
  });

  it("calls anything else a failed schedule", () => {
    expect(sendAtResetFailureTitle(new Error("409"))).toBe(
      SCHEDULE_FAILED_TITLE,
    );
  });
});

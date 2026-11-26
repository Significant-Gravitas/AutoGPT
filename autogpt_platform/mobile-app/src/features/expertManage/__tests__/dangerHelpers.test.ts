import { firePauseLine, isGoneError, resetOutcome } from "../dangerHelpers";

test("says what firing pauses", () => {
  expect(firePauseLine({ schedule_names: [], trigger_names: [] })).toBe("Nothing is scheduled, so nothing pauses.");
  expect(firePauseLine({ schedule_names: ["Daily digest"], trigger_names: [] })).toBe("1 routine pauses: Daily digest.");
  expect(firePauseLine({ schedule_names: ["A", "B"], trigger_names: ["C"] })).toBe("3 routines pause: A, B, C.");
});
test("a reset refused while the expert works can be retried", () => {
  expect(resetOutcome({ status: 409 })).toBe("busy");
  expect(resetOutcome({ status: 503 })).toBe("failed");
  expect(resetOutcome(new Error("offline"))).toBe("failed");
});
test("a 404 means the expert is already gone", () => {
  expect(isGoneError({ status: 404 })).toBe(true);
  expect(isGoneError({ status: 409 })).toBe(false);
  expect(isGoneError(new Error("offline"))).toBe(false);
});

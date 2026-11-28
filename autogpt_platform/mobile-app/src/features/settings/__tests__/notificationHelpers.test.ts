import { BRIEFING_OPTIONS, timezoneLabel, timezoneOffer, withChange } from "../notificationHelpers";

const saved = {
  email: "a@b.co",
  briefing_frequency: "WEEKLY" as const,
  alerts_enabled: true,
  store_verdicts_enabled: false,
  daily_limit: 3,
};

test("a change keeps every field it doesn't touch", () => {
  expect(withChange(saved, { alerts_enabled: false })).toEqual({ ...saved, alerts_enabled: false });
});

test("briefing choices read as on the web", () => {
  expect(BRIEFING_OPTIONS.map((o) => o.label)).toEqual(["Daily", "Weekly", "Monthly", "Off"]);
});

test("offers the phone's time zone only when it differs", () => {
  expect(timezoneOffer("not-set", "Europe/Madrid")).toBe("Europe/Madrid");
  expect(timezoneOffer("Europe/London", "Europe/Madrid")).toBe("Europe/Madrid");
  expect(timezoneOffer("Europe/Madrid", "Europe/Madrid")).toBeNull();
  expect(timezoneOffer("Europe/London", "")).toBeNull();
});

test("an unset time zone reads as such", () => {
  expect(timezoneLabel("not-set")).toBe("Not set");
  expect(timezoneLabel("Europe/Madrid")).toBe("Europe/Madrid");
});

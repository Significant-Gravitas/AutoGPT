import {
  analyticsEnvironment,
  baseProperties,
  canSend,
  focusedRouteNames,
  screenName,
} from "../rules";

test("sends only with consent, a key, and outside development", () => {
  expect(canSend({ consent: "allowed", hasKey: true, isDev: false })).toBe(true);
  expect(canSend({ consent: "unset", hasKey: true, isDev: false })).toBe(false);
  expect(canSend({ consent: "declined", hasKey: true, isDev: false })).toBe(false);
  expect(canSend({ consent: "allowed", hasKey: false, isDev: false })).toBe(false);
  expect(canSend({ consent: "allowed", hasKey: true, isDev: true })).toBe(false);
});

test("names the environment as the tracking plan does", () => {
  expect(analyticsEnvironment("production")).toBe("prod");
  expect(analyticsEnvironment("dev")).toBe("dev");
  expect(analyticsEnvironment(undefined)).toBe("local");
});

test("every event says it came from the mobile app", () => {
  expect(baseProperties("prod", "0.2.0")).toEqual({
    environment: "prod",
    source: "mobile",
    app_version: "0.2.0",
  });
  expect(baseProperties("dev", null)).toEqual({ environment: "dev", source: "mobile" });
});

test("a screen is named by its route, never by an id", () => {
  expect(screenName(["(app)", "chat", "[sessionId]"])).toBe("chat/[sessionId]");
  expect(screenName(["(app)"])).toBe("home");
  expect(screenName(["login"])).toBe("login");
});

test("follows the navigation state down to the screen in front", () => {
  const state = {
    index: 0,
    routes: [
      {
        name: "(app)",
        state: { index: 1, routes: [{ name: "index" }, { name: "files/index" }] },
      },
    ],
  };
  expect(screenName(focusedRouteNames(state))).toBe("files");
  expect(focusedRouteNames(undefined)).toEqual([]);
});

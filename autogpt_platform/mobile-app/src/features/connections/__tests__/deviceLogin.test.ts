import { deviceLoginID, parseDeviceLogin } from "../deviceLogin";

const url =
  "https://app.example/api/proxy/api/integrations/codex/device-login/abc%2F1#state=s&verification_url=https%3A%2F%2Fauth.openai.com%2Fcodex%2Fdevice&user_code=WXYZ-1234";

test("reads the login id, the code and where to enter it", () => {
  expect(parseDeviceLogin(url)).toEqual({
    loginID: "abc/1",
    userCode: "WXYZ-1234",
    verificationURL: "https://auth.openai.com/codex/device",
  });
});

test.each([
  url.replace("user_code=WXYZ-1234", "user_code="),
  url.replace("https%3A%2F%2F", "http%3A%2F%2F"),
  url.replace("https%3A%2F%2Fauth.openai.com%2Fcodex%2Fdevice", "javascript%3Aalert(1)"),
  "https://app.example/elsewhere#user_code=A&verification_url=https%3A%2F%2Fa.example",
  "not a url",
])("refuses %s", (bad) => {
  expect(parseDeviceLogin(bad)).toBeNull();
});

describe("deviceLoginID", () => {
  const cancelURL = "/api/proxy/api/integrations/codex/device-login/abc%2F1/cancel";

  test("reads the id of a sign-in whose page can't be used", () => {
    expect(deviceLoginID(url.replace("user_code=WXYZ-1234", "user_code="), null)).toBe("abc/1");
  });
  test("falls back to the cancel address", () => {
    expect(deviceLoginID("https://app.example/elsewhere", cancelURL)).toBe("abc/1");
  });
  test.each([
    ["not a url", null],
    ["https://app.example/elsewhere", "/api/proxy/elsewhere/cancel"],
    ["https://app.example/elsewhere", "/api/proxy/api/integrations/codex/device-login//cancel"],
  ])("finds none in %s / %s", (loginURL, cancel) => {
    expect(deviceLoginID(loginURL, cancel)).toBeNull();
  });
});

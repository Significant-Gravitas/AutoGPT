import {
  nameFromEmail,
  parseStoredSession,
  readSignInFailure,
  readSignUpOutcome,
  readUser,
  serverSlug,
  sessionCookieHeader,
  storageKey,
  verificationCallbackURL,
} from "../sessionHelpers";

const user = { id: "u1", email: "sam@example.com", name: "Sam", image: null };

describe("storageKey", () => {
  it("keeps one session per server", () => {
    expect(storageKey("https://platform.example.com")).not.toBe(
      storageKey("https://dev.example.com"),
    );
    expect(storageKey("http://localhost:3000")).not.toBe(
      storageKey("http://localhost:3001"),
    );
  });

  it("keeps origins apart that differ only in a character the key store refuses", () => {
    expect(storageKey("http://dev:3000")).not.toBe(storageKey("http://dev_3000"));
  });

  it("uses only characters the phone's key store takes", () => {
    expect(storageKey("http://10.0.2.2:3000/")).toMatch(/^[A-Za-z0-9._-]+$/);
  });
});

describe("parseStoredSession", () => {
  it("reads back what was saved", () => {
    const saved = JSON.stringify({ token: "abc.sig", user });

    expect(parseStoredSession(saved)).toEqual({ token: "abc.sig", user });
  });

  it.each([null, "", "not json", "{}", JSON.stringify({ token: "abc.sig" })])(
    "treats %j as signed out",
    (raw) => {
      expect(parseStoredSession(raw)).toBeNull();
    },
  );
});

describe("readUser", () => {
  it("finds the user in a sign-in answer", () => {
    expect(readUser({ token: "unsigned", user })).toEqual(user);
  });

  it("fills in a missing name and picture", () => {
    expect(readUser({ user: { id: "u1", email: "sam@example.com" } })).toEqual({
      id: "u1",
      email: "sam@example.com",
      name: "",
      image: null,
    });
  });

  it("finds none when the session is gone", () => {
    expect(readUser(null)).toBeNull();
    expect(readUser({ user: null })).toBeNull();
  });
});

describe("readSignInFailure", () => {
  it("keeps the server's status, code and message", async () => {
    const response = Response.json(
      { code: "EMAIL_NOT_VERIFIED", message: "Email not verified" },
      { status: 403 },
    );

    expect(await readSignInFailure(response)).toEqual({
      status: 403,
      code: "EMAIL_NOT_VERIFIED",
      message: "Email not verified",
    });
  });

  it("keeps the status alone when the answer is not JSON", async () => {
    const response = new Response("Bad gateway", { status: 502 });

    expect(await readSignInFailure(response)).toEqual({ status: 502 });
  });
});

describe("sessionCookieHeader", () => {
  it("names the cookie as an http site does", () => {
    expect(sessionCookieHeader("abc.sig", "http://localhost:3000")).toBe(
      "better-auth.session_token=abc.sig",
    );
  });

  it("names the cookie as an https site does", () => {
    expect(sessionCookieHeader("abc.sig", "https://platform.example.com")).toBe(
      "__Secure-better-auth.session_token=abc.sig",
    );
  });

  it("encodes the value once, whether or not it came encoded", () => {
    const expected = "better-auth.session_token=abc.s%2Fi%2Bg%3D";

    expect(sessionCookieHeader("abc.s/i+g=", "http://localhost:3000")).toBe(
      expected,
    );
    expect(
      sessionCookieHeader("abc.s%2Fi%2Bg%3D", "http://localhost:3000"),
    ).toBe(expected);
  });
});

describe("verificationCallbackURL", () => {
  it("sends the link's reader to the page that points back at the app", () => {
    expect(verificationCallbackURL(false)).toBe(
      "/auth/callback?method=email&next=%2Fapp-verified",
    );
  });

  it("carries a marketing refusal in the link", () => {
    expect(verificationCallbackURL(true)).toBe(
      "/auth/callback?method=email&next=%2Fapp-verified&marketing_opt_out=1",
    );
  });
});

describe("nameFromEmail", () => {
  it("is the part before the @, as the web names a new account", () => {
    expect(nameFromEmail("Ada.Lovelace@Northwind.co")).toBe("Ada.Lovelace");
  });
});

describe("readSignUpOutcome", () => {
  const newUser = { id: "u1", email: "ada@northwind.co" };

  it("is signed in when the server hands over a credential", () => {
    expect(readSignUpOutcome({ token: "t", user: newUser }, "t.signed")).toBe(
      "signedIn",
    );
  });

  it("waits for the email when the server made no session", () => {
    expect(readSignUpOutcome({ token: null, user: newUser }, null)).toBe("verify");
  });

  it("names a server without the installed-app contract", () => {
    expect(readSignUpOutcome({ token: "t", user: newUser }, null)).toBe(
      "noAppSession",
    );
    expect(readSignUpOutcome({ token: "t" }, "t.signed")).toBe("noAppSession");
  });
});

describe("serverSlug", () => {
  it("keeps two servers apart", () => {
    expect(serverSlug("http://dev:3000")).not.toBe(serverSlug("http://dev_3000"));
    expect(storageKey("http://dev:3000")).toBe(
      `autogpt.session.${serverSlug("http://dev:3000")}`,
    );
  });
});

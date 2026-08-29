import { beforeEach, describe, expect, it, vi } from "vitest";

const signOutMock = vi.fn();
const cookieDeleteMock = vi.fn();
let cookieJar: { name: string; value: string }[] = [];

vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: {
      signOut: (...args: unknown[]) => signOutMock(...args),
    },
  },
}));

vi.mock("next/headers", () => ({
  cookies: vi.fn(async () => ({
    getAll: () => cookieJar,
    get: (name: string) => cookieJar.find((c) => c.name === name),
    delete: (name: string) => cookieDeleteMock(name),
  })),
}));

import { rollbackSession } from "../rollbackSession";

beforeEach(() => {
  signOutMock.mockReset().mockResolvedValue({ success: true });
  cookieDeleteMock.mockReset();
  cookieJar = [
    { name: "better-auth.session_token", value: "tok.sig" },
    { name: "better-auth.session_data", value: "cached" },
    { name: "theme", value: "dark" },
  ];
});

describe("rollbackSession", () => {
  it("revokes the session using the pending cookie store, not request headers", async () => {
    await rollbackSession();

    expect(signOutMock).toHaveBeenCalledTimes(1);
    const headers = signOutMock.mock.calls[0][0].headers as Headers;
    expect(headers.get("cookie")).toContain(
      "better-auth.session_token=tok.sig",
    );
  });

  it("clears the session cookies but leaves unrelated cookies alone", async () => {
    await rollbackSession();

    expect(cookieDeleteMock).toHaveBeenCalledWith("better-auth.session_token");
    expect(cookieDeleteMock).toHaveBeenCalledWith("better-auth.session_data");
    expect(cookieDeleteMock).not.toHaveBeenCalledWith("theme");
  });

  it("swallows sign-out failures so the caller's error reporting wins", async () => {
    signOutMock.mockRejectedValue(new Error("session backend down"));
    const errorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);

    await expect(rollbackSession()).resolves.toBeUndefined();
    expect(errorSpy).toHaveBeenCalled();
  });
});

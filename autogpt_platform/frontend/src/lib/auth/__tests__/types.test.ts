import { describe, expect, it } from "vitest";
import { mapSessionUser } from "../types";

describe("mapSessionUser", () => {
  it("maps an admin user with role preserved", () => {
    const user = mapSessionUser({
      id: "user-1",
      email: "admin@example.com",
      name: "Admin",
      role: "admin",
      createdAt: new Date("2026-01-02T03:04:05.000Z"),
    });

    expect(user).toEqual({
      id: "user-1",
      email: "admin@example.com",
      role: "admin",
      created_at: "2026-01-02T03:04:05.000Z",
      user_metadata: { name: "Admin", email: "admin@example.com" },
    });
  });

  it("normalizes non-admin roles to authenticated", () => {
    expect(mapSessionUser({ id: "u", email: "a@b.c", role: "user" }).role).toBe(
      "authenticated",
    );
    expect(mapSessionUser({ id: "u", email: "a@b.c", role: null }).role).toBe(
      "authenticated",
    );
  });

  it("passes string createdAt through unchanged", () => {
    const user = mapSessionUser({
      id: "u",
      email: "a@b.c",
      createdAt: "2026-05-06T00:00:00.000Z",
    });

    expect(user.created_at).toBe("2026-05-06T00:00:00.000Z");
  });

  it("leaves user_metadata.name undefined when the user has no name", () => {
    const user = mapSessionUser({ id: "u", email: "a@b.c", name: null });

    expect(user.user_metadata.name).toBeUndefined();
    expect(user.user_metadata.email).toBe("a@b.c");
  });
});

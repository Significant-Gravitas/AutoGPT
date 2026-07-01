import { APIError } from "better-auth/api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getServerSessionMock = vi.fn();
const updateUserMock = vi.fn();
const changeEmailMock = vi.fn();
const getSessionMock = vi.fn();

vi.mock("@/lib/auth/server/getServerSession", () => ({
  getServerSession: () => getServerSessionMock(),
}));

vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: {
      updateUser: (...args: unknown[]) => updateUserMock(...args),
      changeEmail: (...args: unknown[]) => changeEmailMock(...args),
      getSession: (...args: unknown[]) => getSessionMock(...args),
    },
  },
}));

import { GET, PUT } from "../route";

const sessionUser = {
  id: "user-1",
  email: "user@example.com",
  name: "Test User",
  role: "user",
  createdAt: "2026-01-02T03:04:05.000Z",
};

const mappedUser = {
  id: "user-1",
  email: "user@example.com",
  role: "authenticated",
  created_at: "2026-01-02T03:04:05.000Z",
  user_metadata: { name: "Test User", email: "user@example.com" },
};

function makePutRequest(body: string): Request {
  return new Request("http://localhost:3000/api/auth/user", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body,
  });
}

beforeEach(() => {
  getServerSessionMock.mockReset();
  updateUserMock.mockReset();
  changeEmailMock.mockReset();
  getSessionMock.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GET /api/auth/user", () => {
  it("returns the compat-shaped user for an active session", async () => {
    getServerSessionMock.mockResolvedValue({ user: sessionUser });

    const response = await GET();

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ user: mappedUser });
  });

  it("returns 400 with an error when there is no session", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const response = await GET();

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "No active session" });
  });
});

describe("PUT /api/auth/user", () => {
  it("updates the display name via Better Auth and returns the refreshed user", async () => {
    updateUserMock.mockResolvedValue({ status: true });
    getSessionMock.mockResolvedValue({
      user: { ...sessionUser, name: "New Name" },
    });

    const request = makePutRequest(JSON.stringify({ full_name: "New Name" }));
    const response = await PUT(request);

    expect(updateUserMock).toHaveBeenCalledWith({
      body: { name: "New Name" },
      headers: request.headers,
    });
    expect(changeEmailMock).not.toHaveBeenCalled();
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      user: {
        ...mappedUser,
        user_metadata: { ...mappedUser.user_metadata, name: "New Name" },
      },
    });
  });

  it("changes the email via Better Auth when only email is provided", async () => {
    changeEmailMock.mockResolvedValue({ status: true });
    getSessionMock.mockResolvedValue({ user: sessionUser });

    const request = makePutRequest(
      JSON.stringify({ email: "new@example.com" }),
    );
    const response = await PUT(request);

    expect(changeEmailMock).toHaveBeenCalledWith({
      body: { newEmail: "new@example.com" },
      headers: request.headers,
    });
    expect(updateUserMock).not.toHaveBeenCalled();
    expect(response.status).toBe(200);
  });

  it("returns 400 when neither email nor full_name is provided", async () => {
    const response = await PUT(makePutRequest(JSON.stringify({})));

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({
      error: "Email or full_name is required",
    });
    expect(updateUserMock).not.toHaveBeenCalled();
    expect(changeEmailMock).not.toHaveBeenCalled();
  });

  it("returns 400 when the request body is not valid JSON", async () => {
    const response = await PUT(makePutRequest("{not json"));

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "Invalid JSON body" });
  });

  it("passes the Better Auth APIError message through as a 400", async () => {
    updateUserMock.mockRejectedValue(
      new APIError("BAD_REQUEST", {
        message: "Name is too long",
        code: "INVALID_NAME",
      }),
    );

    const response = await PUT(
      makePutRequest(JSON.stringify({ full_name: "New Name" })),
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "Name is too long" });
  });

  it("returns 400 when the session disappears after the update", async () => {
    updateUserMock.mockResolvedValue({ status: true });
    getSessionMock.mockResolvedValue(null);

    const response = await PUT(
      makePutRequest(JSON.stringify({ full_name: "New Name" })),
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: "No active session" });
  });

  it("returns 500 when an unexpected error escapes the auth update", async () => {
    updateUserMock.mockRejectedValue(new Error("db connection lost"));

    const response = await PUT(
      makePutRequest(JSON.stringify({ full_name: "New Name" })),
    );

    expect(response.status).toBe(500);
    expect(await response.json()).toEqual({ error: "Failed to update user" });
  });
});

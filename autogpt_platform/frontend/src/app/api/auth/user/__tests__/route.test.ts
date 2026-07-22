import { beforeEach, describe, expect, it, vi } from "vitest";

const mockUpdateUser = vi.hoisted(() =>
  vi.fn(() => Promise.resolve({ data: { user: { id: "u1" } }, error: null })),
);

vi.mock("@/lib/supabase/server/getServerSupabase", () => ({
  getServerSupabase: () =>
    Promise.resolve({ auth: { updateUser: mockUpdateUser } }),
}));

import { PUT } from "../route";

function makeRequest(body: unknown) {
  return new Request("http://localhost/api/auth/user", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("PUT /api/auth/user", () => {
  beforeEach(() => {
    mockUpdateUser.mockClear();
  });

  it("updates preferred_name metadata", async () => {
    const res = await PUT(makeRequest({ preferred_name: " Rein " }));
    expect(res.status).toBe(200);
    expect(mockUpdateUser).toHaveBeenCalledWith({
      data: { preferred_name: "Rein" },
    });
  });

  it("still updates full_name metadata", async () => {
    const res = await PUT(makeRequest({ full_name: "Jane Doe" }));
    expect(res.status).toBe(200);
    expect(mockUpdateUser).toHaveBeenCalledWith({
      data: { full_name: "Jane Doe" },
    });
  });

  it("still updates email", async () => {
    const res = await PUT(makeRequest({ email: "new@example.com" }));
    expect(res.status).toBe(200);
    expect(mockUpdateUser).toHaveBeenCalledWith({ email: "new@example.com" });
  });

  it("rejects a body without any supported field", async () => {
    const res = await PUT(makeRequest({ something: "else" }));
    expect(res.status).toBe(400);
    expect(mockUpdateUser).not.toHaveBeenCalled();
  });
});

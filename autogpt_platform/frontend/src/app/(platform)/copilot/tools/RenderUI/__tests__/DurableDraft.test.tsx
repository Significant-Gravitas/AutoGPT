import { fireEvent, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { showResult } from "./renderConnected";
import { formView } from "@/lib/openui/__tests__/capability-fixtures";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";

const source = formView(
  'Field("notes", "Notes", "Original", "", false)',
).replace("[form])", '[form], "draft-review")');
const path = "*/api/chat/sessions/connected-session/ui-drafts/draft-review";
describe("durable chat drafts", () => {
  beforeEach(() => {
    sessionStorage.clear();
    localStorage.clear();
  });
  it("prefers a newer server draft over a clean older device cache", async () => {
    const previous = useAuthStore.getState().user;
    useAuthStore.setState({
      user: {
        id: "draft-owner",
        email: "owner@example.com",
        user_metadata: {},
      },
    });
    const key = "copilot-ui:v2:draft-owner:connected-session:draft-review";
    localStorage.setItem(
      key,
      JSON.stringify({
        source,
        revision: 1,
        dirty: false,
        state: {
          review: {
            notes: { value: "Old device copy", componentType: "Field" },
          },
        },
      }),
    );
    server.use(
      http.get(path, () =>
        HttpResponse.json({
          revision: 3,
          source,
          writable: true,
          tool_call_id: "connected-ui",
          state: {
            review: {
              notes: { value: "Newer server copy", componentType: "Field" },
            },
          },
        }),
      ),
    );
    try {
      await showResult(vi.fn(), false, source);
      await waitFor(() =>
        expect((screen.getByLabelText("Notes") as HTMLInputElement).value).toBe(
          "Newer server copy",
        ),
      );
    } finally {
      useAuthStore.setState({ user: previous });
    }
  });
  it("restores the owner-scoped backend draft and saves subsequent edits with its revision", async () => {
    const writes: Record<string, unknown>[] = [];
    server.use(
      http.get(path, () =>
        HttpResponse.json({
          revision: 3,
          source,
          state: {
            review: {
              notes: { value: "Previously saved", componentType: "Field" },
            },
          },
          tool_call_id: "connected-ui",
          writable: true,
        }),
      ),
      http.put(path, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        writes.push(body);
        return HttpResponse.json({
          revision: 4,
          source,
          state: body.state,
          tool_call_id: "connected-ui",
          writable: true,
        });
      }),
    );
    await showResult(vi.fn(), false, source);
    await waitFor(() =>
      expect((screen.getByLabelText("Notes") as HTMLInputElement).value).toBe(
        "Previously saved",
      ),
    );
    fireEvent.change(screen.getByLabelText("Notes"), {
      target: { value: "Changed here" },
    });
    await waitFor(() => expect(writes.length).toBe(1), { timeout: 3000 });
    expect(writes[0].expected_revision).toBe(3);
    expect(JSON.stringify(writes[0].state)).toContain("Changed here");
  });
  it("keeps edits after a conflict and requires an explicit reload", async () => {
    let conflict = false;
    server.use(
      http.get(path, () =>
        HttpResponse.json({
          revision: conflict ? 2 : 1,
          source,
          state: {
            review: {
              notes: {
                value: conflict ? "Other device" : "Original",
                componentType: "Field",
              },
            },
          },
          tool_call_id: "connected-ui",
          writable: true,
        }),
      ),
      http.put(path, () => {
        conflict = true;
        return HttpResponse.json({ detail: "Conflict" }, { status: 409 });
      }),
    );
    await showResult(vi.fn(), false, source);
    fireEvent.change(await screen.findByLabelText("Notes"), {
      target: { value: "My edits" },
    });
    await screen.findByRole(
      "button",
      { name: "Reload saved draft" },
      { timeout: 3000 },
    );
    expect((screen.getByLabelText("Notes") as HTMLInputElement).value).toBe(
      "My edits",
    );
    fireEvent.click(screen.getByRole("button", { name: "Reload saved draft" }));
    await waitFor(() =>
      expect((screen.getByLabelText("Notes") as HTMLInputElement).value).toBe(
        "Other device",
      ),
    );
  });
  it("keeps superseded cards read-only and never requests private drafts on shared chats", async () => {
    let reads = 0,
      writes = 0;
    server.use(
      http.get(path, () => {
        reads++;
        return HttpResponse.json({
          revision: 5,
          source,
          state: {},
          tool_call_id: "newer-ui",
          writable: false,
        });
      }),
      http.put(path, () => {
        writes++;
        return HttpResponse.json({});
      }),
    );
    const result = await showResult(vi.fn(), false, source);
    expect(
      ((await screen.findByLabelText("Notes")) as HTMLInputElement).disabled,
    ).toBe(true);
    expect(screen.getByText(/newer version of this view/)).toBeDefined();
    const privateReads = reads;
    result.unmount();
    await showResult(vi.fn(), true, source);
    expect(reads).toBe(privateReads);
    expect(writes).toBe(0);
  });
});

import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { SharedMemorySection } from "./SharedMemorySection";

const HELD_URL = "http://localhost:3000/api/proxy/api/orgs/org-1/memory/held";
const ORG_URL = "http://localhost:3000/api/proxy/api/orgs/org-1";

const ORG: OrgResponse = {
  id: "org-1",
  name: "Acme",
  slug: "acme",
  avatar_url: null,
  description: null,
  is_personal: false,
  member_count: 3,
  created_at: new Date("2026-01-01T00:00:00Z"),
  memory_hold_buffer: true,
};

function noopSaved() {}

describe("SharedMemorySection", () => {
  it("renders the shared-memory card and review queue for org admins", async () => {
    server.use(
      http.get(HELD_URL, () =>
        HttpResponse.json({ org_id: "org-1", items: [] }, { status: 200 }),
      ),
    );

    render(<SharedMemorySection org={ORG} isAdmin onSaved={noopSaved} />);

    expect(screen.getByTestId("org-shared-memory-section")).toBeDefined();
    expect(
      screen.getByRole("heading", { name: "Shared memory" }),
    ).toBeDefined();
    await waitFor(() =>
      expect(screen.getByTestId("org-memory-review-queue")).toBeDefined(),
    );
  });

  it("renders nothing (and makes no request) for non-admins", () => {
    render(
      <SharedMemorySection org={ORG} isAdmin={false} onSaved={noopSaved} />,
    );

    expect(screen.queryByTestId("org-shared-memory-section")).toBeNull();
  });

  it("reflects the persisted hold-for-review value on the toggle", async () => {
    server.use(
      http.get(HELD_URL, () =>
        HttpResponse.json({ org_id: "org-1", items: [] }, { status: 200 }),
      ),
    );

    render(<SharedMemorySection org={ORG} isAdmin onSaved={noopSaved} />);

    const toggle = screen.getByRole("switch", {
      name: "Hold new memories for review",
    });
    expect(toggle.getAttribute("aria-checked")).toBe("true");
    expect(toggle.hasAttribute("disabled")).toBe(false);
  });

  it("persists memory_hold_buffer when the toggle is switched off", async () => {
    let patchBody: unknown = null;
    server.use(
      http.get(HELD_URL, () =>
        HttpResponse.json({ org_id: "org-1", items: [] }, { status: 200 }),
      ),
      http.patch(ORG_URL, async ({ request }) => {
        patchBody = await request.json();
        return HttpResponse.json(
          { ...ORG, memory_hold_buffer: false },
          { status: 200 },
        );
      }),
    );

    render(<SharedMemorySection org={ORG} isAdmin onSaved={noopSaved} />);

    const toggle = screen.getByRole("switch", {
      name: "Hold new memories for review",
    });
    await userEvent.click(toggle);

    await waitFor(() =>
      expect(patchBody).toEqual({ memory_hold_buffer: false }),
    );
  });
});

import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";
import { OrgAvatarControl } from "./OrgAvatarControl";

const AVATAR_URL = "http://localhost:3000/api/proxy/api/orgs/org-1/avatar";

const ORG: OrgResponse = {
  id: "org-1",
  name: "Acme Corp",
  slug: "acme",
  avatar_url: null,
  description: null,
  is_personal: false,
  member_count: 3,
  created_at: new Date("2026-01-01T00:00:00Z"),
  memory_hold_buffer: true,
};

describe("OrgAvatarControl", () => {
  it("falls back to org initials and offers a Change button to admins", () => {
    render(<OrgAvatarControl org={ORG} isAdmin onSaved={vi.fn()} />);

    expect(screen.getByTestId("org-avatar-initials").textContent).toBe("AC");
    expect(screen.getByRole("button", { name: "Change" })).toBeDefined();
  });

  it("hides the Change button for non-admins", () => {
    render(<OrgAvatarControl org={ORG} isAdmin={false} onSaved={vi.fn()} />);

    expect(screen.getByTestId("org-avatar-initials")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Change" })).toBeNull();
  });

  it("uploads the selected image as multipart and refreshes the org", async () => {
    let uploadedFilename: string | null = null;
    let sawFile = false;
    const onSaved = vi.fn();

    server.use(
      http.post(AVATAR_URL, async ({ request }) => {
        const form = await request.formData();
        const file = form.get("file");
        if (file instanceof File) {
          sawFile = true;
          uploadedFilename = file.name;
        }
        return HttpResponse.json(
          { ...ORG, avatar_url: "https://cdn.test/org-1.png" },
          { status: 200 },
        );
      }),
    );

    render(<OrgAvatarControl org={ORG} isAdmin onSaved={onSaved} />);

    const input = screen.getByLabelText("Upload organization avatar");
    const file = new File(["binary"], "logo.png", { type: "image/png" });
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => expect(sawFile).toBe(true));
    expect(uploadedFilename).toBe("logo.png");
    await waitFor(() => expect(onSaved).toHaveBeenCalled());
  });

  it("surfaces a 400 from the avatar endpoint without calling onSaved", async () => {
    let postCalled = false;
    const onSaved = vi.fn();
    server.use(
      http.post(AVATAR_URL, () => {
        postCalled = true;
        return HttpResponse.json(
          { detail: "Unsupported image type" },
          { status: 400 },
        );
      }),
    );

    render(<OrgAvatarControl org={ORG} isAdmin onSaved={onSaved} />);

    const input = screen.getByLabelText("Upload organization avatar");
    const file = new File(["binary"], "logo.png", { type: "image/png" });
    fireEvent.change(input, { target: { files: [file] } });

    // The request fires, but the failed upload never advances to onSaved.
    await waitFor(() => expect(postCalled).toBe(true));
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("only accepts image files in the picker", () => {
    render(<OrgAvatarControl org={ORG} isAdmin onSaved={vi.fn()} />);

    const input = screen.getByLabelText(
      "Upload organization avatar",
    ) as HTMLInputElement;
    expect(input.accept).toBe("image/png,image/jpeg,image/webp,image/gif");
  });
});

import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { OrgAccessView } from "../components/OrgAccessView/OrgAccessView";

const mockToast = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

const GRANT = {
  user_id: "user-1",
  email: "maya@acme.test",
  name: "Maya Chen",
  granted_at: "2026-11-02T10:00:00Z",
  granted_by_user_id: "admin-1",
  granted_by_email: "admin@agpt.co",
  note: "Team plan trial",
};

function page(grants: (typeof GRANT)[]) {
  return { grants, total: grants.length, page: 1, page_size: 25 };
}

function accessStatus(hasAccess: boolean) {
  return {
    user_id: GRANT.user_id,
    email: GRANT.email,
    grant: hasAccess ? GRANT : null,
    inherited_from: [],
    has_access: hasAccess,
  };
}

beforeEach(() => {
  mockToast.mockReset();
});

describe("Organization access", () => {
  it("lists the accounts an admin gave organizations", async () => {
    server.use(
      http.get("*/api/admin/org-access", () =>
        HttpResponse.json(page([GRANT])),
      ),
    );

    render(<OrgAccessView />);

    const row = await screen.findByTestId("org-access-row");
    expect(within(row).getByText("maya@acme.test")).toBeDefined();
    expect(within(row).getByText(/Team plan trial/)).toBeDefined();
    expect(within(row).getByText(/by admin@agpt\.co/)).toBeDefined();
  });

  it("turns organizations on for the account an email names", async () => {
    let posted: unknown = null;
    server.use(
      http.get("*/api/admin/org-access", () =>
        HttpResponse.json(page(posted ? [GRANT] : [])),
      ),
      http.post("*/api/admin/org-access/by-email", async ({ request }) => {
        posted = await request.json();
        return HttpResponse.json(accessStatus(true));
      }),
    );

    render(<OrgAccessView />);
    expect(
      await screen.findByText("No account has organizations yet."),
    ).toBeDefined();
    fireEvent.change(screen.getByLabelText("Account email"), {
      target: { value: " maya@acme.test " },
    });
    fireEvent.change(screen.getByLabelText("Note"), {
      target: { value: "Team plan trial" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Turn on" }));

    await waitFor(() =>
      expect(posted).toEqual({
        email: "maya@acme.test",
        note: "Team plan trial",
      }),
    );
    expect(await screen.findByTestId("org-access-row")).toBeDefined();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "success" }),
    );
  });

  it("shows why an email could not be turned on", async () => {
    server.use(
      http.get("*/api/admin/org-access", () => HttpResponse.json(page([]))),
      http.post("*/api/admin/org-access/by-email", () =>
        HttpResponse.json(
          { detail: "No account uses that email" },
          { status: 404 },
        ),
      ),
    );

    render(<OrgAccessView />);
    fireEvent.change(await screen.findByLabelText("Account email"), {
      target: { value: "nobody@acme.test" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Turn on" }));

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't turn organizations on",
          variant: "destructive",
        }),
      ),
    );
  });

  it("turns organizations off once the admin confirms", async () => {
    let removed: string | null = null;
    server.use(
      http.get("*/api/admin/org-access", () =>
        HttpResponse.json(page(removed ? [] : [GRANT])),
      ),
      http.delete("*/api/admin/org-access/users/:userId", ({ params }) => {
        removed = String(params.userId);
        return HttpResponse.json(accessStatus(false));
      }),
    );

    render(<OrgAccessView />);
    const row = await screen.findByTestId("org-access-row");
    fireEvent.click(within(row).getByRole("button", { name: "Turn off" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Turn off" }));

    await waitFor(() => expect(removed).toBe("user-1"));
    expect(
      await screen.findByText("No account has organizations yet."),
    ).toBeDefined();
  });
});

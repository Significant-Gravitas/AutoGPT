import { getGetV2AdminListPendingSkillSubmissionsMockHandler200 } from "@/app/api/__generated__/endpoints/admin/admin.msw";
import type { SkillSubmission } from "@/app/api/__generated__/models/skillSubmission";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, test, vi } from "vitest";
import { AdminSkillSubmissions } from "../components/AdminSkillSubmissions";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/admin/marketplace",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const pending: SkillSubmission = {
  skill_listing_version_id: "version-1",
  slug: "brand-voice-guide",
  name: "brand-voice-guide",
  description: "Write in a consistent brand voice.",
  categories: ["content"],
  required_providers: ["google"],
  version: 1,
  status: "PENDING",
  review_comments: null,
  is_live: false,
};

describe("admin skill submissions", () => {
  test("shows what a reviewer needs to judge the submission", async () => {
    server.use(
      getGetV2AdminListPendingSkillSubmissionsMockHandler200([pending]),
    );

    render(<AdminSkillSubmissions />);

    expect(await screen.findByTestId("admin-skill-submissions")).toBeDefined();
    expect(
      await screen.findByText("Write in a consistent brand voice."),
    ).toBeDefined();
    expect(
      screen.getByText(/brand-voice-guide · content · works with google/),
    ).toBeDefined();
  });

  test("approving posts the verdict for that submission", async () => {
    let reviewed: { url: string; body: unknown } | null = null;
    server.use(
      getGetV2AdminListPendingSkillSubmissionsMockHandler200([pending]),
      http.post(
        "*/api/store/admin/skills/submissions/:id/review",
        async ({ request, params }) => {
          reviewed = { url: String(params.id), body: await request.json() };
          return HttpResponse.json({ ...pending, status: "APPROVED" });
        },
      ),
    );
    render(<AdminSkillSubmissions />);

    await userEvent.click(await screen.findByTestId("approve-version-1"));

    await waitFor(() => expect(reviewed).not.toBeNull());
    expect(reviewed!.url).toBe("version-1");
    expect((reviewed!.body as { is_approved: boolean }).is_approved).toBe(true);
  });

  test("rejecting posts the opposite verdict", async () => {
    let approved: boolean | null = null;
    server.use(
      getGetV2AdminListPendingSkillSubmissionsMockHandler200([pending]),
      http.post(
        "*/api/store/admin/skills/submissions/:id/review",
        async ({ request }) => {
          const body = (await request.json()) as { is_approved: boolean };
          approved = body.is_approved;
          return HttpResponse.json({ ...pending, status: "REJECTED" });
        },
      ),
    );
    render(<AdminSkillSubmissions />);

    await userEvent.click(await screen.findByTestId("reject-version-1"));

    await waitFor(() => expect(approved).toBe(false));
  });

  test("says the queue is empty rather than rendering an empty list", async () => {
    server.use(getGetV2AdminListPendingSkillSubmissionsMockHandler200([]));

    render(<AdminSkillSubmissions />);

    expect(
      await screen.findByText(/No skill submissions are waiting/),
    ).toBeDefined();
    expect(screen.queryByTestId("admin-skill-submissions")).toBeNull();
  });
});

import {
  getGetV2ListStoreCategoriesMockHandler200,
  getPostV2PublishSkillToMarketplaceMockHandler201,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { getGetV1ListProvidersMockHandler200 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";

// Radix marks the page inert while a dialog is open, which userEvent refuses
// to click through by default.
const user = () => userEvent.setup({ pointerEventsCheck: 0 });
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { PublishSkillButton } from "../PublishSkillButton";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/library/skills",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const categories = [
  { value: "content", label: "Content", description: "Writing and media" },
  { value: "sales", label: "Sales", description: "Outreach and CRM" },
];

describe("PublishSkillButton", () => {
  beforeEach(() => {
    server.use(
      getGetV2ListStoreCategoriesMockHandler200(categories),
      getGetV1ListProvidersMockHandler200([
        { name: "google", description: "Google" },
      ]),
      getPostV2PublishSkillToMarketplaceMockHandler201(),
    );
  });

  test("will not submit until a category is chosen", async () => {
    const u = user();
    render(<PublishSkillButton skillName="brand-voice-guide" />);

    await u.click(screen.getByTestId("skill-publish-button"));

    const submit = await screen.findByTestId("skill-publish-submit");
    expect(submit.hasAttribute("disabled")).toBe(true);
  });

  test("submits the skill with its category and integrations", async () => {
    const u = user();
    let body: unknown = null;
    server.use(
      http.post("*/api/store/skills/submissions", async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(
          {
            skill_listing_version_id: "v1",
            slug: "brand-voice-guide",
            name: "brand-voice-guide",
            description: "d",
            categories: ["content"],
            required_providers: ["google"],
            version: 1,
            status: "PENDING",
            review_comments: null,
            is_live: false,
          },
          { status: 201 },
        );
      }),
    );
    render(<PublishSkillButton skillName="brand-voice-guide" />);
    await u.click(screen.getByTestId("skill-publish-button"));

    await u.click(await screen.findByText("Pick a category"));
    await u.click(await screen.findByText("Content"));
    await u.click(await screen.findByText("Google"));
    await u.click(screen.getByTestId("skill-publish-submit"));

    expect(await screen.findByTestId("skill-publish-submitted")).toBeDefined();
    await waitFor(() =>
      expect(body).toEqual({
        skill_name: "brand-voice-guide",
        categories: ["content"],
        required_providers: ["google"],
      }),
    );
  });

  test("says the published copy is a snapshot, not a live link", async () => {
    const u = user();
    render(<PublishSkillButton skillName="brand-voice-guide" />);
    await u.click(screen.getByTestId("skill-publish-button"));
    await u.click(await screen.findByText("Pick a category"));
    await u.click(await screen.findByText("Content"));
    await u.click(screen.getByTestId("skill-publish-submit"));

    const done = await screen.findByTestId("skill-publish-submitted");
    expect(done.textContent).toMatch(/copy/i);
    expect(done.textContent).toMatch(/until you publish again/i);
  });
});

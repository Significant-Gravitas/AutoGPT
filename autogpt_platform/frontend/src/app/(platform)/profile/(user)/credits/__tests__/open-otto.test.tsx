import { describe, expect, test, vi } from "vitest";

import { getGetV2GetCopilotUsageMockHandler } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import CreditsPage from "../page";

const routerPush = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush, replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/profile/credits",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => true,
}));

describe("legacy credits page usage", () => {
  test("Open Otto goes to the chat", async () => {
    server.use(
      getGetV2GetCopilotUsageMockHandler({
        daily: { percent_used: 40, resets_at: new Date("2026-01-16") },
        weekly: { percent_used: 10, resets_at: new Date("2026-01-20") },
      }),
    );

    render(<CreditsPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Open Otto" }));

    expect(routerPush).toHaveBeenCalledWith("/copilot");
  });
});

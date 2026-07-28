import {
  getHireExpertMockHandler,
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "hire-experts" ? true : actual.useGetFlag(flag as never),
  };
});

const mariaTemplate: Expert = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  is_template: true,
  source_template_id: null,
  is_archived: false,
  workflows: [],
};

const hiredMaria: Expert = {
  ...mariaTemplate,
  id: "expert-maria",
  is_template: false,
  source_template_id: "template-maria",
};

function renderMarketplace() {
  return render(
    <>
      <MainMarkeplacePage />
      <Toaster />
    </>,
  );
}

describe("Marketplace ExpertsSection", () => {
  test("renders experts section and hires from the profile sheet", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    await userEvent.click(await screen.findByText("Maria"));
    await userEvent.click(await screen.findByRole("button", { name: "Hire" }));

    expect(await screen.findByText("Maria joined your team")).toBeDefined();
    expect(await screen.findByText("Chat with Maria")).toBeDefined();
  });

  test("hired template shows hired state", async () => {
    server.use(
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([hiredMaria]),
    );

    renderMarketplace();

    expect(await screen.findByText("Meet the AI Experts")).toBeDefined();
    expect(await screen.findByText("Hired")).toBeDefined();
  });
});

import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { render } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

vi.mock("@/components/contextual/CronScheduler/cron-scheduler-dialog", () => ({
  CronExpressionDialog: () => null,
}));

vi.mock("./components/ThumbnailImages", () => ({
  ThumbnailImages: () => <div data-testid="thumbnail-images-mock" />,
}));

import { AgentInfoStep } from "../AgentInfoStep";

describe("AgentInfoStep", () => {
  const baseProps = {
    onBack: vi.fn(),
    onSuccess: vi.fn(),
    selectedAgentId: "graph-1",
    selectedAgentVersion: 1,
    initialData: undefined,
    isMarketplaceUpdate: false,
  };

  it("renders the listing form with the three accordion sections", async () => {
    render(<AgentInfoStep {...baseProps} />);

    expect(await screen.findByText("Build the store listing")).toBeDefined();
    expect(screen.getByText("Listing basics")).toBeDefined();
    expect(screen.getByText("Thumbnails")).toBeDefined();
    expect(screen.getByText("Experience details")).toBeDefined();
    expect(
      screen.getByRole("button", { name: /submit for review/i }),
    ).toBeDefined();
  });

  it("calls onBack when Back is clicked", async () => {
    const onBack = vi.fn();
    render(<AgentInfoStep {...baseProps} onBack={onBack} />);
    await screen.findByText("Build the store listing");
    fireEvent.click(screen.getByRole("button", { name: /back/i }));
    expect(onBack).toHaveBeenCalled();
  });

  it("renders the marketplace-update copy when isMarketplaceUpdate is true", async () => {
    render(<AgentInfoStep {...baseProps} isMarketplaceUpdate />);
    expect(await screen.findByText("Describe the update")).toBeDefined();
  });

  it("mounts cleanly when the submissions API would error (handler installed)", async () => {
    server.use(
      http.post("http://localhost:3000/api/proxy/api/store/submissions", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    render(<AgentInfoStep {...baseProps} />);
    expect(await screen.findByText("Build the store listing")).toBeDefined();
  });
});

import {
  getGetExpertComputerMockHandler,
  getStartExpertDesktopMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ComputerInfo } from "@/app/api/__generated__/models/computerInfo";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ExpertComputerSection } from "../ExpertComputerSection";

const computer: ComputerInfo = {
  owner_kind: "expert",
  owner_id: "expert-1",
  e2b_active: true,
  shell: {
    kind: "shell",
    sandbox_id: "sb-shell",
    state: "paused",
    started_at: new Date("2026-09-05T12:00:00Z"),
    cpu_count: 2,
    memory_mb: 512,
    template_id: "base",
    mounts_attached: true,
  },
  desktop: null,
  mounts: {
    "/home/user/workspace": "autogpt-expert-expert-1",
    "/home/user/shared": "autogpt-user-u1",
  },
  workspace_path: "/home/user/workspace",
  shared_path: "/home/user/shared",
};

describe("ExpertComputerSection", () => {
  it("shows the expert's boxes and volumes without waking anything", async () => {
    server.use(getGetExpertComputerMockHandler(computer));

    render(
      <ExpertComputerSection expertId="expert-1" expertName="Maria" enabled />,
    );

    expect(await screen.findByText("Maria's computer")).toBeDefined();
    expect(screen.getByText("Suspended")).toBeDefined();
    expect(screen.getByText("2 vCPU · 512 MiB")).toBeDefined();
    expect(screen.getByText("/home/user/shared")).toBeDefined();
    expect(screen.getByText("autogpt-expert-expert-1")).toBeDefined();
    expect(screen.getByRole("button", { name: "Start desktop" })).toBeDefined();
  });

  it("starts the desktop and embeds its stream", async () => {
    server.use(
      getGetExpertComputerMockHandler(computer),
      getStartExpertDesktopMockHandler({
        kind: "desktop_stream",
        url: "https://6080-sbx.e2b.app/vnc.html?autoconnect=true",
        provider: "e2b",
        sandbox_id: "sbx-desktop",
        requires_auth: false,
      }),
    );

    render(
      <ExpertComputerSection expertId="expert-1" expertName="Maria" enabled />,
    );

    await userEvent.click(
      await screen.findByRole("button", { name: "Start desktop" }),
    );

    await waitFor(() =>
      expect(
        screen.getByTitle("Interactive desktop (sbx-desktop)"),
      ).toBeDefined(),
    );
  });

  it("explains when sandboxes are not configured", async () => {
    server.use(
      getGetExpertComputerMockHandler({
        ...computer,
        e2b_active: false,
        shell: null,
        mounts: {},
      }),
    );

    render(
      <ExpertComputerSection expertId="expert-1" expertName="Maria" enabled />,
    );

    expect(
      await screen.findByText(
        "Cloud sandboxes are not configured on this deployment.",
      ),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Start desktop" }),
    ).toHaveProperty("disabled", true);
  });
});

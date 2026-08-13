import { getGetV1ListCredentialsMockHandler } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import {
  getGetV1ListExecutionSchedulesForAUserMockHandler,
  getListCopilotFollowupSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { useCopilotModal } from "../../../useCopilotModal";
import { CopilotModals } from "../CopilotModals";

function Harness() {
  const { openModal } = useCopilotModal();
  return (
    <>
      <button onClick={() => openModal("skills")}>open-skills</button>
      <button onClick={() => openModal("scheduled")}>open-scheduled</button>
      <button onClick={() => openModal("integrations")}>
        open-integrations
      </button>
      <CopilotModals />
    </>
  );
}

describe("CopilotModals", () => {
  beforeEach(() => {
    useCopilotUIStore.setState({ initialPrompt: null });
    server.use(
      getListCopilotSkillsMockHandler([]),
      getListCopilotFollowupSchedulesMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([]),
      getGetV1ListCredentialsMockHandler([]),
    );
  });

  afterEach(() => {
    server.resetHandlers();
  });

  test("no modal renders by default", () => {
    render(<Harness />);
    expect(screen.queryByText("AutoPilot skills")).toBeNull();
    expect(screen.queryByText("Scheduled")).toBeNull();
    expect(screen.queryByText("Third Party Integrations")).toBeNull();
  });

  test("opens the skills modal with header actions and empty state", async () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("open-skills"));

    expect(await screen.findByText("AutoPilot skills")).toBeDefined();
    expect(await screen.findByTestId("skills-empty")).toBeDefined();
    expect(screen.getByTestId("skill-new-button")).toBeDefined();
    expect(screen.getByTestId("skill-upload-button")).toBeDefined();
  });

  test("New skill closes the modal and prefills the composer store", async () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("open-skills"));

    fireEvent.click(await screen.findByTestId("skill-new-button"));

    await vi.waitFor(() => {
      expect(useCopilotUIStore.getState().initialPrompt).toContain(
        "I want to teach you a new skill",
      );
    });
    await vi.waitFor(() => {
      expect(screen.queryByText("AutoPilot skills")).toBeNull();
    });
  });

  test("opens the scheduled modal and New scheduled task prefills the store", async () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("open-scheduled"));

    expect(await screen.findByTestId("followups-empty")).toBeDefined();
    fireEvent.click(screen.getByTestId("schedule-new-button"));

    await vi.waitFor(() => {
      expect(useCopilotUIStore.getState().initialPrompt).toContain(
        "I want to create a new scheduled task",
      );
    });
  });

  test("opens the integrations modal with the Connect Service action", async () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("open-integrations"));

    expect(await screen.findByText("Third Party Integrations")).toBeDefined();
    expect(
      (await screen.findAllByText("Connect Service")).length,
    ).toBeGreaterThan(0);
  });
});

import {
  getGetV1ListCredentialsMockHandler,
  getGetV1ListProvidersMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import {
  getGetV1ListExecutionSchedulesForAUserMockHandler,
  getListCopilotFollowupSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { useCopilotModal } from "../../../useCopilotModal";
import { CopilotModals } from "../CopilotModals";

function Harness() {
  const { openModal } = useCopilotModal();

  function openSkills() {
    openModal("skills");
  }

  function openScheduled() {
    openModal("scheduled");
  }

  function openIntegrations() {
    openModal("integrations");
  }

  function openConnect() {
    openModal("connect");
  }

  return (
    <>
      <button onClick={openSkills}>open-skills</button>
      <button onClick={openScheduled}>open-scheduled</button>
      <button onClick={openIntegrations}>open-integrations</button>
      <button onClick={openConnect}>open-connect</button>
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

  test("opens the connect dialog directly, without the credentials list", async () => {
    server.use(
      getGetV1ListProvidersMockHandler([
        {
          name: "github",
          description: "Issues and PRs",
          supported_auth_types: ["oauth2", "api_key"],
        },
      ]),
    );
    render(<Harness />);
    fireEvent.click(screen.getByText("open-connect"));

    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("Issues and PRs")).toBeDefined();
    expect(screen.queryByText("No integration connected")).toBeNull();
  });

  test("renders the connect dialog from a ?modal=connect deep link", async () => {
    render(
      <NuqsTestingAdapter searchParams="?modal=connect">
        <Harness />
      </NuqsTestingAdapter>,
    );

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Connect a service")).toBeDefined();
  });

  test("closing the connect dialog clears the modal query param", async () => {
    const onUrlUpdate = vi.fn();
    render(
      <NuqsTestingAdapter
        searchParams="?modal=connect"
        onUrlUpdate={onUrlUpdate}
      >
        <Harness />
      </NuqsTestingAdapter>,
    );

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("button", { name: "Close" }));

    await vi.waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
    await vi.waitFor(() => {
      expect(
        onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("modal"),
      ).toBeNull();
      expect(onUrlUpdate).toHaveBeenCalled();
    });
  });
});

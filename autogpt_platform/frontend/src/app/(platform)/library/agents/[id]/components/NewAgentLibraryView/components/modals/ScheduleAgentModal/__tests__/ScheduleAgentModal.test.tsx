import {
  getPostV1CreateExecutionScheduleMockHandler,
  getPostV1CreateExecutionScheduleResponseMock,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { ScheduleCreationRequest } from "@/app/api/__generated__/models/scheduleCreationRequest";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { ScheduleAgentModal } from "../ScheduleAgentModal";

const agent = {
  id: "lib-1",
  graph_id: "graph-xyz",
  graph_version: 1,
  name: "My agent",
  recommended_schedule_cron: null,
} as unknown as LibraryAgent;

function captureSentCrons() {
  const crons: string[] = [];
  server.use(
    getPostV1CreateExecutionScheduleMockHandler(async ({ request }) => {
      const body = (await request.json()) as ScheduleCreationRequest;
      crons.push(body.cron);
      return getPostV1CreateExecutionScheduleResponseMock({ cron: body.cron });
    }),
  );
  return crons;
}

async function openCustomInterval(user: ReturnType<typeof userEvent.setup>) {
  render(
    <ScheduleAgentModal
      isOpen
      onClose={vi.fn()}
      agent={agent}
      inputValues={{}}
      inputCredentials={{}}
    />,
  );
  await user.click(await screen.findByRole("combobox", { name: "Repeats" }));
  await user.click(await screen.findByRole("option", { name: "Custom" }));
  return screen.getByRole<HTMLInputElement>("spinbutton", { name: "Every" });
}

async function chooseUnit(
  user: ReturnType<typeof userEvent.setup>,
  unit: "Minutes" | "Hours" | "Days",
) {
  await user.click(screen.getByRole("combobox", { name: "Interval" }));
  await user.click(await screen.findByRole("option", { name: unit }));
}

function scheduleButton() {
  return screen.getByRole("button", { name: "Schedule" });
}

describe("ScheduleAgentModal custom interval (#15275)", () => {
  test.each([
    ["Minutes", "60", "0 * * * *"],
    ["Hours", "24", "0 0 * * *"],
    ["Days", "30", "0 9 */30 * *"],
    ["Minutes", "45", "*/45 * * * *"],
  ] as const)(
    "every %s %s sends %s, a cron the scheduler accepts",
    async (unit, value, cron) => {
      const user = userEvent.setup();
      const crons = captureSentCrons();
      const field = await openCustomInterval(user);
      await chooseUnit(user, unit);

      await user.clear(field);
      await user.type(field, value);
      await user.click(scheduleButton());

      await waitFor(() => expect(crons).toEqual([cron]));
    },
  );

  test("keeps the field as typed after it is cleared", async () => {
    const user = userEvent.setup();
    const crons = captureSentCrons();
    const field = await openCustomInterval(user);

    await user.clear(field);
    expect(field.value).toBe("");
    await user.type(field, "45");
    expect(field.value).toBe("45");
    await user.click(scheduleButton());

    await waitFor(() => expect(crons).toEqual(["*/45 * * * *"]));
  });

  test("flags an interval above the unit's range as it is typed, without rewriting it, and sends nothing until it is fixed", async () => {
    const user = userEvent.setup();
    const crons = captureSentCrons();
    const field = await openCustomInterval(user);
    await chooseUnit(user, "Days");

    await user.clear(field);
    await user.type(field, "31");
    expect(field.value).toBe("31");
    expect(screen.getByText("Enter 1 to 30 days")).toBeDefined();
    expect(scheduleButton()).toHaveProperty("disabled", true);
    await user.click(scheduleButton());

    await user.clear(field);
    await user.type(field, "30");
    expect(screen.queryByText("Enter 1 to 30 days")).toBeNull();
    await user.click(scheduleButton());

    await waitFor(() => expect(crons).toEqual(["0 9 */30 * *"]));
  });

  test("an emptied field is flagged only once the user leaves it", async () => {
    const user = userEvent.setup();
    const field = await openCustomInterval(user);

    await user.clear(field);
    expect(screen.queryByText("Enter 1 to 60 minutes")).toBeNull();
    await user.tab();

    expect(screen.getByText("Enter 1 to 60 minutes")).toBeDefined();
    expect(field.value).toBe("");
    expect(scheduleButton()).toHaveProperty("disabled", true);
  });

  test("flags a prefilled interval that is already out of range", async () => {
    render(
      <ScheduleAgentModal
        isOpen
        onClose={vi.fn()}
        agent={{ ...agent, recommended_schedule_cron: "*/90 * * * *" }}
        inputValues={{}}
        inputCredentials={{}}
      />,
    );

    const field = await screen.findByRole<HTMLInputElement>("spinbutton", {
      name: "Every",
    });
    expect(field.value).toBe("90");
    expect(await screen.findByText("Enter 1 to 60 minutes")).toBeDefined();
    expect(field.getAttribute("aria-invalid")).toBe("true");
    expect(scheduleButton()).toHaveProperty("disabled", true);
  });

  test("a unit change re-checks the interval instead of changing it", async () => {
    const user = userEvent.setup();
    const crons = captureSentCrons();
    const field = await openCustomInterval(user);

    await user.clear(field);
    await user.type(field, "45");
    await chooseUnit(user, "Hours");

    expect(field.value).toBe("45");
    expect(screen.getByText("Enter 1 to 24 hours")).toBeDefined();
    expect(scheduleButton()).toHaveProperty("disabled", true);
    await user.click(scheduleButton());

    await user.clear(field);
    await user.type(field, "12");
    await user.click(scheduleButton());
    await waitFor(() => expect(crons).toEqual(["0 */12 * * *"]));
  });
});

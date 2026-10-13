import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";
import {
  launchSchedule,
  rehearsalSchedule,
} from "@/lib/openui/__tests__/schedule-fixtures";
import { render } from "@/tests/integrations/test-utils";
import { OpenUI } from "@/components/organisms/OpenUI/OpenUI";

describe("editable schedules in the conversation", () => {
  beforeEach(() => sessionStorage.clear());
  it("distinguishes a capacity shortfall from whole tasks that remain unscheduled", async () => {
    const source = `root = Workspace("Museum visits", "Illustrative", [form])
form = Form("day", "Choose a reduction", [Schedule("visits", "Requested visits", [{label:"Afternoon",windows:[{start:"13:00",end:"16:00"}]}], [{id:"first",title:"Museum one",minutes:75,done:false,splittable:false},{id:"transfer1",title:"Transfer",minutes:20,done:false,splittable:false},{id:"second",title:"Museum two",minutes:75,done:false,splittable:false},{id:"transfer2",title:"Second transfer",minutes:20,done:false,splittable:false},{id:"third",title:"Museum three",minutes:75,done:false,splittable:false}])], "Discuss plan", "Help reduce this plan")`;
    await showResult(vi.fn(), false, source);
    expect(await screen.findByText("85 min over available time")).toBeDefined();
    expect(
      screen.getByText("95 min unscheduled · 10 min unused"),
    ).toBeDefined();
  });
  it("keeps a streamed schedule usable as windows and tasks arrive", async () => {
    function pending(source: string) {
      return (
        <OpenUI
          source={source}
          isStreaming
          disabled
          revision={0}
          onAction={vi.fn()}
        />
      );
    }
    const cuts = ['"Monday"', '"09:00"', '"Research"', '"Design"'];
    const view = render(pending('root = Workspace("Plan", "", [plan])'));
    for (const fragment of cuts) {
      const end = launchSchedule.indexOf(fragment) + fragment.length;
      view.rerender(pending(launchSchedule.slice(0, end)));
      await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    }
    view.rerender(pending(launchSchedule));
    expect(await screen.findByText("60 min spare")).toBeDefined();
    expect(
      (screen.getByLabelText("Design minutes") as HTMLInputElement).disabled,
    ).toBe(true);
  });
  it("recalculates when durations change, preserves edits on reload and submits typed tasks", async () => {
    const send = vi.fn();
    const first = await showResult(send, false, launchSchedule);
    expect(await screen.findByText("60 min spare")).toBeDefined();
    fireEvent.change(screen.getByLabelText("Design minutes"), {
      target: { value: "420" },
    });
    expect(
      await screen.findByText("60 min unscheduled · 0 min unused"),
    ).toBeDefined();
    first.unmount();
    await showResult(send, false, launchSchedule);
    expect(
      ((await screen.findByLabelText("Design minutes")) as HTMLInputElement)
        .value,
    ).toBe("420");
    fireEvent.change(screen.getByLabelText("Design minutes"), {
      target: { value: "abc" },
    });
    expect(
      await screen.findByText(
        "Enter valid durations to calculate the schedule.",
      ),
    ).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Discuss this plan" }));
    expect(send).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Design minutes"), {
      target: { value: "300" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Discuss this plan" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain('"minutes": 300');
    expect(send.mock.calls[0][0]).toContain('"work_buffer_minutes": 60');
  });
  it("detects a departure conflict after reordering and preserves reported completion", async () => {
    await showResult(vi.fn(), false, rehearsalSchedule);
    expect(await screen.findByText("20 min spare")).toBeDefined();
    expect(
      (screen.getByLabelText("Completed Setup") as HTMLInputElement).checked,
    ).toBe(true);
    fireEvent.click(
      screen.getByRole("button", { name: "Move Rehearsal B later" }),
    );
    expect(
      await screen.findByText("Cannot finish before its deadline"),
    ).toBeDefined();
    expect(screen.queryByText("20 min spare")).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: "Move Rehearsal B earlier" }),
    );
    expect(await screen.findByText("20 min spare")).toBeDefined();
  });
  it("keeps shared schedules read-only", async () => {
    await showResult(vi.fn(), true, launchSchedule);
    expect(
      (await screen.findByLabelText("Design minutes")).hasAttribute("disabled"),
    ).toBe(true);
    expect(
      screen
        .getByRole("button", { name: "Move Design earlier" })
        .hasAttribute("disabled"),
    ).toBe(true);
  });
  it("keeps an invalid completed duration local without losing the completed task", async () => {
    const send = vi.fn();
    await showResult(send, false, launchSchedule);
    fireEvent.change(await screen.findByLabelText("Design minutes"), {
      target: { value: "abc" },
    });
    fireEvent.click(screen.getByLabelText("Completed Design"));
    expect(await screen.findByText("360 min spare")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Discuss this plan" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).not.toContain("abc");
    expect(send.mock.calls[0][0]).toContain('"title": "Design"');
    expect(send.mock.calls[0][0]).toContain('"done": true');
    expect(send.mock.calls[0][0]).toContain('"work_required_minutes": 480');
  });
  it("submits every task and derived value in a maximum-size form", async () => {
    const send = vi.fn();
    const names = Array.from({ length: 6 }, (_, index) => `plan${index}`);
    const days = [
      { label: "Monday", windows: [{ start: "09:00", end: "10:00" }] },
    ];
    const tasks = [
      {
        id: "review",
        title: "Review",
        minutes: 30,
        done: false,
        splittable: false,
      },
    ];
    const source = [
      'root = Workspace("Six plans", "Supplied windows", [plans])',
      `plans = Form("plans", "Plans", [${names.join(", ")}], "Discuss plans", "Compare these plans")`,
      ...names.map(
        (name) =>
          `${name} = Schedule("${name}", "${name}", ${JSON.stringify(days)}, ${JSON.stringify(tasks)})`,
      ),
    ].join("\n");
    await showResult(send, false, source);
    fireEvent.click(
      await screen.findByRole("button", { name: "Discuss plans" }),
    );
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    for (const name of names) {
      expect(send.mock.calls[0][0]).toContain(`"${name}": [`);
      expect(send.mock.calls[0][0]).toContain(`"${name}_buffer_minutes": 30`);
      expect(send.mock.calls[0][0]).toContain(`"${name}_feasible": true`);
    }
  });
});

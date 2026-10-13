import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";
import { formView } from "@/lib/openui/__tests__/capability-fixtures";

describe("typed planning controls in chat", () => {
  beforeEach(() => sessionStorage.clear());
  it("keeps hidden repeatable fields out of submission and restores their local draft", async () => {
    const send = vi.fn();
    await showResult(
      send,
      false,
      formView(
        'ToggleField("overnight", "Stay overnight", true)',
        'FieldGroup("guests", "Guests", [{key:"email",label:"Guest email",type:"email",required:true}], [{id:"guest",cells:{email:"bad"}}], true, true, {field:"overnight",equals:true})',
      ),
    );
    fireEvent.click(await screen.findByRole("button", { name: "Continue" }));
    expect(send).not.toHaveBeenCalled();
    fireEvent.click(screen.getByLabelText("Stay overnight"));
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    const fields = JSON.parse(
      send.mock.calls[0][0].split("Submitted values:\n")[1],
    );
    expect(fields.guests).toEqual([]);
    expect(send.mock.calls[0][0]).not.toContain('"bad"');
    fireEvent.click(screen.getByLabelText("Stay overnight"));
    expect(
      (screen.getByLabelText(/Guest email/) as HTMLInputElement).value,
    ).toBe("bad");
  });
  it("requires a daylight-saving occurrence before sending a resolved instant", async () => {
    const send = vi.fn();
    await showResult(
      send,
      false,
      formView(
        'DateTimeField("appointment", "Meeting", {local:"2026-11-01T01:30",timeZone:"America/Los_Angeles",occurrence:""}, true)',
      ),
    );
    fireEvent.click(await screen.findByRole("button", { name: "Continue" }));
    expect(send).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Meeting occurrence"), {
      target: { value: "later" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(send.mock.calls[0][0]).toContain("2026-11-01T09:30:00.000Z");
  });
  it("sends the entire edited document and never performs an external write", async () => {
    const send = vi.fn();
    const text = "Long draft. ".repeat(400) + "THE END";
    await showResult(
      send,
      false,
      formView(
        'DocumentEditor("copy", "Review announcement", "Initial", "Initial", "announcement.md", true)',
      ),
    );
    fireEvent.change(await screen.findByLabelText("Current draft"), {
      target: { value: text },
    });
    expect(send).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(
      JSON.parse(send.mock.calls[0][0].split("Submitted values:\n")[1]).copy,
    ).toBe(text);
  });
  it("blocks starting a dependent task until its prerequisite is complete", async () => {
    const tasks = [
      {
        id: "research",
        title: "Research",
        owner: "",
        due: "",
        status: "todo",
        dependsOn: [],
      },
      {
        id: "write",
        title: "Write",
        owner: "",
        due: "",
        status: "todo",
        dependsOn: ["research"],
      },
    ];
    const send = vi.fn();
    await showResult(
      send,
      false,
      formView(
        `TaskBoard("tasks", "Publication", ${JSON.stringify(tasks)}, true)`,
      ),
    );
    const write = await screen.findByLabelText("Status for Write");
    fireEvent.change(write, { target: { value: "doing" } });
    expect((write as HTMLSelectElement).value).toBe("todo");
    fireEvent.change(screen.getByLabelText("Status for Research"), {
      target: { value: "done" },
    });
    fireEvent.change(screen.getByLabelText("Status for Write"), {
      target: { value: "doing" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(
      JSON.parse(send.mock.calls[0][0].split("Submitted values:\n")[1]).tasks[1]
        .status,
    ).toBe("doing");
  });
});

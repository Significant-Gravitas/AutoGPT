import { describe, expect, it } from "vitest";
import { screen, fireEvent, waitFor, within } from "@testing-library/react";
import { render } from "@/tests/integrations/test-utils";
import { OpenUILab } from "../components/OpenUILab/OpenUILab";

describe("OpenUI lab", () => {
  it("renders the sample through OpenUI and switches between UI and source", async () => {
    render(<OpenUILab standalone liveAvailable={false} />);
    expect(await screen.findByText("Your agents, at a glance")).toBeDefined();
    expect(screen.getByText("1,284")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Source" }));
    expect(screen.getByLabelText("OpenUI source").textContent).toContain(
      "Workspace(",
    );
    fireEvent.click(screen.getByRole("button", { name: "Interactive" }));
    expect(screen.getByText("1,284")).toBeDefined();
  });

  it("turns a generated action into a new workspace", async () => {
    render(<OpenUILab standalone liveAvailable={false} />);
    const workspace = within(
      screen.getByRole("region", { name: "Generated workspace" }),
    );
    fireEvent.click(
      await workspace.findByRole("button", { name: "Investigate failed runs" }),
    );
    expect(
      await screen.findByText(
        "A closer look at failed runs",
        {},
        { timeout: 5000 },
      ),
    ).toBeDefined();
    expect(
      await screen.findByText("Retry rate limits with backoff"),
    ).toBeDefined();
  });

  it("filters generated table data and preserves the results when sorting", async () => {
    render(<OpenUILab standalone liveAvailable={false} />);
    fireEvent.change(
      await screen.findByLabelText("Filter Your top performers"),
      { target: { value: "researcher" } },
    );
    expect(screen.getByText("Lead researcher")).toBeDefined();
    expect(screen.queryByText("Content studio")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Runs/ }));
    expect(screen.getByText("486")).toBeDefined();
  });

  it("does not submit Enter while an IME is composing", async () => {
    render(<OpenUILab standalone liveAvailable={false} />);
    const input = screen.getByLabelText("Message");
    fireEvent.change(input, { target: { value: "Investigate failed runs" } });
    fireEvent.keyDown(input, { key: "Enter", isComposing: true, keyCode: 229 });
    expect(screen.queryByText("Rendering the sample…")).toBeNull();
    expect((input as HTMLTextAreaElement).value).toBe(
      "Investigate failed runs",
    );
  });

  it("renders an editable brief and feeds its values into a sample plan", async () => {
    render(<OpenUILab standalone liveAvailable={false} />);
    fireEvent.click(screen.getByRole("button", { name: /Campaign planner/ }));
    const audience = await screen.findByLabelText(
      "Audience",
      {},
      { timeout: 5000 },
    );
    fireEvent.change(audience, { target: { value: "Independent bookshops" } });
    fireEvent.click(screen.getByRole("button", { name: "Source" }));
    fireEvent.click(screen.getByRole("button", { name: "Interactive" }));
    expect((screen.getByLabelText("Audience") as HTMLInputElement).value).toBe(
      "Independent bookshops",
    );
    await waitFor(() =>
      expect(
        screen
          .getByRole("button", { name: "Build my plan" })
          .hasAttribute("disabled"),
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole("button", { name: "Build my plan" }));
    expect(
      await screen.findByText(
        /A launch plan for Independent bookshops/,
        {},
        { timeout: 5000 },
      ),
    ).toBeDefined();
    const task = await screen.findByRole("checkbox", {
      name: /Define the offer/,
    });
    fireEvent.click(task);
    expect((task as HTMLInputElement).checked).toBe(true);
  });

  it("explains the sample limitation instead of pretending to generate arbitrary requests", async () => {
    render(<OpenUILab standalone liveAvailable={false} />);
    fireEvent.change(screen.getByLabelText("Message"), {
      target: { value: "Design an underwater observatory" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(await screen.findByRole("alert")).toBeDefined();
    expect(screen.getByRole("alert").textContent).toContain("Live AI");
    expect(screen.getByText("1,284")).toBeDefined();
  });
});

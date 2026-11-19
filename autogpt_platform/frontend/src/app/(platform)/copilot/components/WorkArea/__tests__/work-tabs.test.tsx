import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@/tests/integrations/test-utils";
import { TAB_ICONS, WorkTabs } from "../WorkTabs";

const LONG = "25 look-alike design partners for the spring launch";

function renderTabs() {
  return render(
    <WorkTabs
      tabs={[
        { key: "task:task-1", title: LONG, icon: TAB_ICONS.task },
        { key: "profile", title: "Profile", icon: TAB_ICONS.profile },
      ]}
      activeKey="profile"
      onSelect={vi.fn()}
      onClose={vi.fn()}
      isWide={false}
      onToggleWidth={vi.fn()}
      onFold={vi.fn()}
    />,
  );
}

describe("Sidebar tabs", () => {
  it("shows a long tab name in full when the tab is pointed at", async () => {
    renderTabs();

    await userEvent.hover(screen.getByRole("tab", { name: LONG }));

    expect((await screen.findByRole("tooltip")).textContent).toBe(LONG);
  });

  it("shows the full name when focus moves onto the tab", async () => {
    renderTabs();

    act(() => screen.getByRole("tab", { name: LONG }).focus());

    expect((await screen.findByRole("tooltip")).textContent).toBe(LONG);
  });
});

import { render, screen } from "@/tests/integrations/test-utils";
import { SidebarProvider } from "@/components/ui/sidebar";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { useGlobalSearchStore } from "@/app/(platform)/components/GlobalSearchModal/useGlobalSearchStore";
import { SidebarSearch } from "../SidebarSearch";

function renderSearch() {
  return render(
    <SidebarProvider>
      <SidebarSearch />
    </SidebarProvider>,
  );
}

describe("SidebarSearch", () => {
  beforeEach(() => {
    useGlobalSearchStore.setState({ isOpen: false });
  });

  it("renders a Search menu button", () => {
    renderSearch();
    expect(screen.getByText("Search")).toBeDefined();
  });

  it("opens the global search palette when clicked", async () => {
    const user = userEvent.setup();
    renderSearch();

    expect(useGlobalSearchStore.getState().isOpen).toBe(false);
    await user.click(screen.getByText("Search"));
    expect(useGlobalSearchStore.getState().isOpen).toBe(true);
  });
});

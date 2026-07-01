import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { SidebarProvider } from "@/components/ui/sidebar";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { RecentChatItem } from "../RecentChatItem";

interface Session {
  id: string;
  title?: string | null;
  source_platform?: string | null;
  is_processing?: boolean | null;
  updated_at: string;
}

const baseSession: Session = {
  id: "s1",
  title: "My chat",
  updated_at: "2026-06-30T00:00:00Z",
};

function makeProps(
  overrides: Partial<React.ComponentProps<typeof RecentChatItem>> = {},
) {
  return {
    session: baseSession,
    isActive: false,
    isEditing: false,
    editingTitle: "",
    onEditingTitleChange: vi.fn(),
    onSubmitRename: vi.fn(),
    onCancelRename: vi.fn(),
    isExporting: false,
    isDeleting: false,
    chatSharingEnabled: false,
    onRename: vi.fn(),
    onExport: vi.fn(),
    onShare: vi.fn(),
    onDelete: vi.fn(),
    ...overrides,
  };
}

function renderItem(props: React.ComponentProps<typeof RecentChatItem>) {
  return render(
    <SidebarProvider>
      <RecentChatItem {...props} />
    </SidebarProvider>,
  );
}

function openActions() {
  fireEvent.pointerDown(screen.getByRole("button", { name: /chat actions/i }), {
    button: 0,
  });
}

describe("RecentChatItem — display", () => {
  it("links to the session and shows its title", () => {
    renderItem(makeProps());
    const link = screen.getByRole("link", { name: /my chat/i });
    expect(link.getAttribute("href")).toBe("/copilot?sessionId=s1");
  });

  it("falls back to 'Untitled chat' when the title is empty", () => {
    renderItem(makeProps({ session: { ...baseSession, title: null } }));
    expect(screen.getByText("Untitled chat")).toBeDefined();
  });

  it("shows a platform logo for external-origin chats", () => {
    renderItem(
      makeProps({
        session: { ...baseSession, source_platform: "discord" },
      }),
    );
    expect(screen.getByAltText("Discord")).toBeDefined();
  });
});

describe("RecentChatItem — editing mode", () => {
  it("renders an input and submits on Enter", () => {
    const onSubmitRename = vi.fn();
    renderItem(
      makeProps({ isEditing: true, editingTitle: "New title", onSubmitRename }),
    );

    const input = screen.getByLabelText("Rename chat");
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSubmitRename).toHaveBeenCalledWith("s1");
  });

  it("cancels on Escape", () => {
    const onCancelRename = vi.fn();
    renderItem(makeProps({ isEditing: true, onCancelRename }));

    fireEvent.keyDown(screen.getByLabelText("Rename chat"), { key: "Escape" });
    expect(onCancelRename).toHaveBeenCalled();
  });

  it("submits on blur when no key already resolved the edit", () => {
    const onSubmitRename = vi.fn();
    renderItem(makeProps({ isEditing: true, onSubmitRename }));

    fireEvent.blur(screen.getByLabelText("Rename chat"));
    expect(onSubmitRename).toHaveBeenCalledWith("s1");
  });

  it("forwards typed input via onEditingTitleChange", () => {
    const onEditingTitleChange = vi.fn();
    renderItem(makeProps({ isEditing: true, onEditingTitleChange }));

    fireEvent.change(screen.getByLabelText("Rename chat"), {
      target: { value: "abc" },
    });
    expect(onEditingTitleChange).toHaveBeenCalledWith("abc");
  });
});

describe("RecentChatItem — actions menu", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("invokes onRename when Rename is chosen", async () => {
    const onRename = vi.fn();
    renderItem(makeProps({ onRename }));

    openActions();
    fireEvent.click(await screen.findByRole("menuitem", { name: /rename/i }));
    expect(onRename).toHaveBeenCalledWith("s1", "My chat");
  });

  it("invokes onExport when Export chat is chosen", async () => {
    const onExport = vi.fn();
    renderItem(makeProps({ onExport }));

    openActions();
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /export chat/i }),
    );
    expect(onExport).toHaveBeenCalledWith("s1", "My chat");
  });

  it("invokes onDelete when Delete chat is chosen", async () => {
    const onDelete = vi.fn();
    renderItem(makeProps({ onDelete }));

    openActions();
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /delete chat/i }),
    );
    expect(onDelete).toHaveBeenCalledWith("s1", "My chat");
  });

  it("hides the Share action when sharing is disabled", async () => {
    renderItem(makeProps({ chatSharingEnabled: false }));

    openActions();
    await screen.findByRole("menuitem", { name: /rename/i });
    expect(screen.queryByRole("menuitem", { name: /share chat/i })).toBeNull();
  });

  it("shows and triggers the Share action when sharing is enabled", async () => {
    const onShare = vi.fn();
    renderItem(makeProps({ chatSharingEnabled: true, onShare }));

    openActions();
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /share chat/i }),
    );
    expect(onShare).toHaveBeenCalledWith("s1");
  });

  it("shows an exporting label while an export is in flight", async () => {
    renderItem(makeProps({ isExporting: true }));

    openActions();
    expect(await screen.findByText(/exporting/i)).toBeDefined();
  });
});

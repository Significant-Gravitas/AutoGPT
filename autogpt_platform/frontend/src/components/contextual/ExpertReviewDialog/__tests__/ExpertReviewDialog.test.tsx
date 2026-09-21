import type { ExpertPackagePreview } from "@/app/api/__generated__/models/expertPackagePreview";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { ExpertReviewDialog } from "../ExpertReviewDialog";

const preview: ExpertPackagePreview = {
  manifest: {
    format_version: 1,
    identity: {
      name: "Maria",
      role: "Marketing Strategist",
      tagline: "Grows your brand while you sleep",
    },
  },
  avatar_kind: "file",
  skills: [
    {
      slug: "brand-voice",
      name: "Brand voice",
      description: "Keeps every draft on-brand.",
      files: [
        { path: "SKILL.md", size_bytes: 12 },
        { path: "tone.md", size_bytes: 9 },
      ],
    },
  ],
  workflows: [
    {
      index: 0,
      name: "Calendar",
      source: "store",
      schedule_cron: "40 7 * * *",
    },
    { index: 1, name: "Draft Poster", source: "graph" },
  ],
};

const onClose = vi.fn();
const onConfirm = vi.fn();

function renderDialog(
  overrides: Partial<Parameters<typeof ExpertReviewDialog>[0]> = {},
) {
  return render(
    <ExpertReviewDialog
      mode="import"
      open
      preview={preview}
      isSubmitting={false}
      onClose={onClose}
      onConfirm={onConfirm}
      {...overrides}
    />,
  );
}

function primaryButton() {
  return screen.getByRole("button", { name: "Import expert" });
}

beforeEach(() => {
  onClose.mockReset();
  onConfirm.mockReset();
});

describe("ExpertReviewDialog", () => {
  test("shows what the file holds", async () => {
    renderDialog();

    expect(await screen.findByLabelText("Name")).toHaveProperty(
      "value",
      "Maria",
    );
    expect(
      screen.getByText(
        "Marketing Strategist · Grows your brand while you sleep",
      ),
    ).toBeDefined();
    expect(screen.getByText("2 files")).toBeDefined();
    expect(screen.getByText("Marketplace")).toBeDefined();
    expect(screen.getByText("From file")).toBeDefined();
    expect(screen.getByText("Every day at 07:40")).toBeDefined();
  });

  test("a removed skill travels as a slug, and an undo takes it back", async () => {
    renderDialog();

    await userEvent.click(
      await screen.findByRole("button", { name: "Remove Brand voice" }),
    );
    await userEvent.click(primaryButton());
    expect(onConfirm).toHaveBeenCalledWith(
      expect.objectContaining({ removed_skill_slugs: ["brand-voice"] }),
    );

    await userEvent.click(
      screen.getByRole("button", { name: "Keep Brand voice" }),
    );
    await userEvent.click(primaryButton());
    expect(onConfirm).toHaveBeenLastCalledWith(
      expect.objectContaining({ removed_skill_slugs: [] }),
    );
  });

  test("a removed workflow travels as its index, and drops its schedule", async () => {
    renderDialog();

    await userEvent.click(
      await screen.findByRole("button", { name: "Remove Calendar" }),
    );
    await userEvent.click(primaryButton());

    expect(onConfirm).toHaveBeenCalledWith(
      expect.objectContaining({
        removed_workflow_indices: [0],
        workflows: [],
      }),
    );
  });

  test("a packaged schedule starts on and can be turned off", async () => {
    renderDialog();

    const schedule = await screen.findByRole("switch", { name: /Schedule/ });
    expect(schedule.getAttribute("data-state")).toBe("checked");

    await userEvent.click(schedule);
    await userEvent.click(primaryButton());

    expect(onConfirm).toHaveBeenCalledWith(
      expect.objectContaining({
        workflows: [{ index: 0, schedule_enabled: false }],
      }),
    );
  });

  test("an empty name blocks the import", async () => {
    renderDialog();

    await userEvent.clear(await screen.findByLabelText("Name"));

    expect(screen.getByText("Give this expert a name.")).toBeDefined();
    expect(primaryButton().hasAttribute("disabled")).toBe(true);
  });

  test("errors block the import; warnings only say so", async () => {
    const issue = [{ code: "cap", message: "Team is full." }];
    const { unmount } = renderDialog({
      preview: { ...preview, warnings: issue },
    });
    expect(await screen.findByRole("status")).toHaveProperty(
      "textContent",
      "Team is full.",
    );
    expect(primaryButton().hasAttribute("disabled")).toBe(false);
    unmount();

    renderDialog({ preview: { ...preview, errors: issue } });
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      "Team is full.",
    );
    expect(primaryButton().hasAttribute("disabled")).toBe(true);
  });

  test("two identical issues render as two rows with distinct keys", async () => {
    // Two workflows sharing a name produce two issues with the same code and
    // the same message. React still renders both on a first mount when their
    // keys collide, so the assertion that matters is that it did not have to
    // warn about it — a colliding key is what drops one of them on re-render.
    const repeated = [
      { code: "unresolved", message: "'Digest' is not on this marketplace." },
      { code: "unresolved", message: "'Digest' is not on this marketplace." },
    ];
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});

    renderDialog({ preview: { ...preview, warnings: repeated } });

    const banner = await screen.findByRole("status");
    expect(banner.querySelectorAll("li").length).toBe(2);
    expect(consoleError.mock.calls.flat().join(" ")).not.toContain("same key");
    consoleError.mockRestore();
  });

  // Only the publish route can tell whether an agent without a stored listing
  // was published later, so the dialog names it as the admin's own agent and
  // lets the route decide instead of calling it unpublished.
  test("publishing leaves an agent's marketplace status to the server", async () => {
    renderDialog({ mode: "publish" });

    expect(await screen.findByText("Your agent")).toBeDefined();
    expect(screen.queryByText("From file")).toBeNull();
    expect(screen.queryByText(/Publish these agents/)).toBeNull();

    const publish = screen.getByRole("button", { name: "Publish" });
    expect(publish.hasAttribute("disabled")).toBe(false);
    await userEvent.click(publish);
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  // The publish route builds the package from the stored expert and takes no
  // edits, so a control here could only lie about what reaches the marketplace.
  test("publish mode confirms and never edits", async () => {
    renderDialog({ mode: "publish" });

    expect(await screen.findByText("Maria")).toBeDefined();
    expect(screen.queryByLabelText("Name")).toBeNull();
    expect(
      screen.queryAllByRole("button", { name: /^(Remove|Keep) / }),
    ).toEqual([]);
    expect(screen.queryAllByRole("switch")).toEqual([]);
    expect(screen.getByText("Schedule · Every day at 07:40")).toBeDefined();
  });

  test("Escape closes and Enter never submits", async () => {
    renderDialog();

    await userEvent.type(await screen.findByLabelText("Name"), "{Enter}");
    expect(onConfirm).not.toHaveBeenCalled();

    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  test("nothing closes the dialog while the request is in flight", async () => {
    renderDialog({ isSubmitting: true });

    const cancel = await screen.findByRole("button", { name: "Cancel" });
    expect(cancel.hasAttribute("disabled")).toBe(true);

    await userEvent.keyboard("{Escape}");
    expect(onClose).not.toHaveBeenCalled();
  });
});

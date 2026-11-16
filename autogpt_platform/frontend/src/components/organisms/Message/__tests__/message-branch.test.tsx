import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  MessageBranch,
  MessageBranchContent,
  MessageBranchNext,
  MessageBranchPage,
  MessageBranchPrevious,
  MessageBranchSelector,
} from "../Message";

function renderBranches(
  replies: string[],
  onBranchChange?: (index: number) => void,
) {
  return render(
    <MessageBranch onBranchChange={onBranchChange}>
      <MessageBranchContent>
        {replies.map((reply) => (
          <p key={reply}>{reply}</p>
        ))}
      </MessageBranchContent>
      <MessageBranchSelector from="assistant">
        <MessageBranchPrevious />
        <MessageBranchPage />
        <MessageBranchNext />
      </MessageBranchSelector>
    </MessageBranch>,
  );
}

describe("MessageBranch", () => {
  afterEach(() => {
    cleanup();
  });

  it("pages through the replies from the button group", async () => {
    const onBranchChange = vi.fn();
    renderBranches(["First reply", "Second reply"], onBranchChange);

    const group = await screen.findByRole("group");
    expect(group.textContent).toContain("1 of 2");
    expect(screen.getByText("First reply").parentElement?.className).toContain(
      "block",
    );

    fireEvent.click(screen.getByRole("button", { name: "Next branch" }));
    await waitFor(() => expect(group.textContent).toContain("2 of 2"));
    expect(screen.getByText("Second reply").parentElement?.className).toContain(
      "block",
    );

    fireEvent.click(screen.getByRole("button", { name: "Next branch" }));
    await waitFor(() => expect(group.textContent).toContain("1 of 2"));

    fireEvent.click(screen.getByRole("button", { name: "Previous branch" }));
    await waitFor(() => expect(group.textContent).toContain("2 of 2"));
    expect(onBranchChange.mock.calls).toEqual([[1], [0], [1]]);
  });

  it("hides the button group when there is a single reply", () => {
    renderBranches(["Only reply"]);

    expect(screen.getByText("Only reply")).toBeTruthy();
    expect(screen.queryByRole("group")).toBeNull();
  });
});

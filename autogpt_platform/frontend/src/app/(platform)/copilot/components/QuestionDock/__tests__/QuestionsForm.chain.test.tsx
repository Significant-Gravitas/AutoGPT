import { act, cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ClarifyingQuestion } from "../../../tools/clarifying-questions";
import {
  ChainActionsContext,
  type ChainActionEntry,
  type ChainActions,
} from "../../ToolChain/chainActions";
import { QuestionsForm } from "../QuestionDock";

function makeQuestions(): ClarifyingQuestion[] {
  return [{ question: "Which region?", keyword: "region" }];
}

function setup() {
  const register = vi.fn<(entry: ChainActionEntry) => void>();
  const unregister = vi.fn<(id: string) => void>();
  const chainActions: ChainActions = { register, unregister };

  function ui(questions: ClarifyingQuestion[], dockId = "dock-1") {
    return (
      <ChainActionsContext.Provider value={chainActions}>
        <QuestionsForm dockId={dockId} questions={questions} />
      </ChainActionsContext.Provider>
    );
  }

  const view = render(ui(makeQuestions()));
  return { register, unregister, ui, view };
}

function lastEntry(register: {
  mock: { calls: [ChainActionEntry][] };
}): ChainActionEntry {
  return register.mock.calls[register.mock.calls.length - 1][0];
}

describe("QuestionsForm inside a tool chain", () => {
  afterEach(cleanup);

  it("registers once and ignores a fresh but identical questions array", () => {
    const { register, unregister, ui, view } = setup();
    expect(register).toHaveBeenCalledTimes(1);

    // Upstream rebuilds the questions on every render — a new reference with
    // the same content must not re-register (that fed the update loop).
    view.rerender(ui(makeQuestions()));
    view.rerender(ui(makeQuestions()));

    expect(register).toHaveBeenCalledTimes(1);
    expect(unregister).not.toHaveBeenCalled();
  });

  it("updates the entry on answer without unregistering first", () => {
    const { register, unregister } = setup();

    act(() => {
      lastEntry(register).questions?.onAnswer("region", "Europe");
    });

    expect(register).toHaveBeenCalledTimes(2);
    const entry = lastEntry(register);
    expect(entry.ready).toBe(true);
    expect(entry.questions?.answers).toEqual({ region: "Europe" });
    expect(entry.buildMessage()).toContain("Europe");
    expect(unregister).not.toHaveBeenCalled();
  });

  it("unregisters when skipped", () => {
    const { register, unregister } = setup();

    act(() => {
      lastEntry(register).questions?.onSkip();
    });

    expect(unregister).toHaveBeenCalledWith("dock-1");
    expect(register).toHaveBeenCalledTimes(1);
  });

  it("unregisters on unmount", () => {
    const { unregister, view } = setup();

    view.unmount();

    expect(unregister).toHaveBeenCalledWith("dock-1");
  });
});

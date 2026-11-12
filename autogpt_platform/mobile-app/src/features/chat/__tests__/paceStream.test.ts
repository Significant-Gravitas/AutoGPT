import type { UIMessageChunk } from "ai";

import { cutPoint, takeTick } from "../paceStream";

function delta(text: string, id = "t1"): UIMessageChunk {
  return { type: "text-delta", id, delta: text };
}

function drain(queue: UIMessageChunk[]) {
  const ticks: UIMessageChunk[][] = [];
  while (queue.length > 0) ticks.push(takeTick(queue));
  return ticks;
}

function textOf(chunks: UIMessageChunk[]) {
  return chunks.map((chunk) => (chunk.type === "text-delta" ? chunk.delta : "")).join("");
}

describe("takeTick", () => {
  it("lets a burst out over several ticks without losing or reordering text", () => {
    const text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    const ticks = drain([delta(text)]);
    expect(ticks.length).toBeGreaterThan(5);
    expect(ticks.map(textOf).join("")).toBe(text);
  });

  it("lets most of a large burst out within about half a second", () => {
    const text = "word ".repeat(2000);
    const queue = [delta(text)];
    let shown = "";
    for (let tick = 0; tick < 25; tick += 1) shown += textOf(takeTick(queue));
    expect(shown.length).toBeGreaterThanOrEqual(text.length * 0.9);
  });

  it("keeps other chunks in place behind the text before them", () => {
    const queue: UIMessageChunk[] = [
      { type: "text-start", id: "t1" },
      delta("hello there, how are you doing today?"),
      { type: "text-end", id: "t1" },
      { type: "finish" },
    ];
    const order = drain(queue)
      .flat()
      .map((chunk) => chunk.type)
      .filter((type, index, all) => type !== all[index - 1]);
    expect(order).toEqual(["text-start", "text-delta", "text-end", "finish"]);
  });

  it("joins pieces of the same text part into one chunk per tick", () => {
    const tick = takeTick([delta("a"), delta("b"), delta("c")]);
    expect(tick).toEqual([delta("abc")]);
  });
});

describe("cutPoint", () => {
  it("ends on a space close to the budget", () => {
    expect(cutPoint("hello world again", 3)).toBe(6);
  });

  it("cuts at the budget when no space is near", () => {
    expect(cutPoint("a".repeat(40), 5)).toBe(5);
  });

  it("never splits an emoji", () => {
    const text = `ab${"😀"}cd`;
    const cut = cutPoint(text, 3);
    expect(text.slice(0, cut)).toBe(`ab${"😀"}`);
  });
});

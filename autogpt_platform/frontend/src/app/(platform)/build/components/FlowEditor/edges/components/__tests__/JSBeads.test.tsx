import { Profiler } from "react";
import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getLengthOfPathInPixels } from "../../helpers";
import { JSBeads } from "../JSBeads";

vi.mock("../../helpers", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../helpers")>();
  return {
    ...actual,
    getLengthOfPathInPixels: vi.fn((path: string) =>
      path === LONG_PATH ? 200 : 100,
    ),
  };
});

const SHORT_PATH = "M0,0 L100,0";
const LONG_PATH = "M0,0 L200,0";
const FRAME_MS = 16;

interface Props {
  beadUp: number;
  beadDown: number;
  edgePath?: string;
}

let commits = 0;

function Beads({ beadUp, beadDown, edgePath = SHORT_PATH }: Props) {
  return (
    <svg>
      <Profiler
        id="beads"
        onRender={() => {
          commits += 1;
        }}
      >
        <JSBeads
          beadUp={beadUp}
          beadDown={beadDown}
          edgePath={edgePath}
          beadsKey="beads-edge-1"
        />
      </Profiler>
    </svg>
  );
}

function circles(container: HTMLElement) {
  return Array.from(container.querySelectorAll("circle"));
}

function advance(ms: number) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

// One act per frame, so React commits each frame's updates as a browser would.
function advanceFrames(count: number) {
  for (let frame = 0; frame < count; frame++) advance(FRAME_MS);
}

beforeEach(() => {
  commits = 0;
  vi.mocked(getLengthOfPathInPixels).mockClear();
  vi.useFakeTimers({
    toFake: ["requestAnimationFrame", "cancelAnimationFrame"],
  });
  // happy-dom has no SVG geometry: a straight path along the x axis.
  const pathPrototype = Object.getPrototypeOf(
    document.createElementNS("http://www.w3.org/2000/svg", "path"),
  );
  pathPrototype.getPointAtLength = (length: number) => ({ x: length, y: 0 });
  pathPrototype.getTotalLength = function (this: SVGPathElement) {
    return this.getAttribute("d") === LONG_PATH ? 200 : 100;
  };
});

afterEach(() => {
  vi.useRealTimers();
});

describe("JSBeads", () => {
  it("moves a bead along the edge and removes it on arrival, without a render per frame", () => {
    const { container } = render(<Beads beadUp={1} beadDown={1} />);
    expect(circles(container)).toHaveLength(1);

    advanceFrames(10);
    const x = Number(circles(container)[0].getAttribute("cx"));
    expect(x).toBeGreaterThan(0);
    expect(x).toBeLessThan(90);

    advanceFrames(60);
    expect(circles(container)).toHaveLength(0);
    // Mount, bead created, bead removed: not one commit per animation frame.
    expect(commits).toBeLessThanOrEqual(3);
  });

  it("parks a bead one diameter before the end until its node takes it", () => {
    const { container, rerender } = render(<Beads beadUp={1} beadDown={0} />);

    advance(1_000);
    expect(circles(container)).toHaveLength(1);
    expect(circles(container)[0].getAttribute("cx")).toBe("90");
    expect(circles(container)[0].getAttribute("cy")).toBe("0");

    rerender(<Beads beadUp={0} beadDown={0} />);
    advance(FRAME_MS);
    expect(circles(container)).toHaveLength(0);
  });

  it("measures each distinct edge path once and follows the edge when it moves", () => {
    const { container, rerender } = render(<Beads beadUp={1} beadDown={0} />);
    advance(1_000);

    rerender(<Beads beadUp={1} beadDown={0} />);
    rerender(<Beads beadUp={1} beadDown={0} />);
    expect(getLengthOfPathInPixels).toHaveBeenCalledTimes(1);

    rerender(<Beads beadUp={1} beadDown={0} edgePath={LONG_PATH} />);
    rerender(<Beads beadUp={1} beadDown={0} edgePath={LONG_PATH} />);
    expect(getLengthOfPathInPixels).toHaveBeenCalledTimes(2);
    expect(circles(container)[0].getAttribute("cx")).toBe("180");
  });

  it("does not measure an edge that carries no beads", () => {
    const { rerender } = render(<Beads beadUp={0} beadDown={0} />);
    rerender(<Beads beadUp={0} beadDown={0} edgePath={LONG_PATH} />);
    advance(FRAME_MS * 5);

    expect(getLengthOfPathInPixels).not.toHaveBeenCalled();
  });
});

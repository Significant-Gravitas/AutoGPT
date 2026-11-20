import fs from "fs";
import path from "path";
import {
  act,
  createContext,
  memo,
  useContext,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const SCRIPT = fs.readFileSync(
  path.resolve(__dirname, "../../../public/devtools/render-counter.js"),
  "utf8",
);

interface Report {
  commits: number;
  totalRenders: number;
  components: { key: string; count: number }[];
  causes: { key: string; count: number }[];
}

interface Hook {
  onCommitFiberRoot(rendererID: number, root: unknown): void;
}

type TestWindow = Window & {
  __renderCounter?: {
    start(): void;
    stop(): void;
    reset(): void;
    report(): Report;
  };
  __REACT_DEVTOOLS_GLOBAL_HOOK__?: Hook;
};

const testWindow = window as TestWindow;

function loadCounter() {
  new Function(SCRIPT)();
  return testWindow.__renderCounter!;
}

// React DOM is already loaded by the test setup, before the hook existed, so
// it never injected. Hand each commit to the hook the way React would.
function commit(container: HTMLElement) {
  const key = Object.keys(container).find((name) =>
    name.startsWith("__reactContainer$"),
  );
  const hostRoot = (
    container as unknown as Record<string, { stateNode: unknown }>
  )[key!];
  testWindow.__REACT_DEVTOOLS_GLOBAL_HOOK__!.onCommitFiberRoot(
    1,
    hostRoot.stateNode,
  );
}

function causes(report: Report) {
  return Object.fromEntries(report.causes.map((c) => [c.key, c.count]));
}

const ThemeContext = createContext("light");
let setCount: Dispatch<SetStateAction<number>> = () => undefined;
let setTheme: Dispatch<SetStateAction<string>> = () => undefined;

function Leaf() {
  return <span>leaf</span>;
}

const MemoLabel = memo(function Label({ value }: { value: number }) {
  return <span>{value}</span>;
});

const MemoStatic = memo(function Static() {
  return <Leaf />;
});

function ThemeReader() {
  return <span>{useContext(ThemeContext)}</span>;
}

const MemoThemeBox = memo(function ThemeBox() {
  return <ThemeReader />;
});

function Counter() {
  const [count, update] = useState(0);
  setCount = update;
  return (
    <div>
      <Leaf />
      <MemoLabel value={Math.floor(count / 2)} />
      <MemoStatic />
    </div>
  );
}

function App() {
  const [theme, update] = useState("light");
  setTheme = update;
  return (
    <ThemeContext.Provider value={theme}>
      <Counter />
      <MemoThemeBox />
    </ThemeContext.Provider>
  );
}

describe("render-counter.js", () => {
  let container: HTMLElement;
  let root: Root;

  beforeEach(() => {
    delete testWindow.__renderCounter;
    delete testWindow.__REACT_DEVTOOLS_GLOBAL_HOOK__;
    container = document.createElement("div");
    document.body.appendChild(container);
  });

  afterEach(() => {
    act(() => root?.unmount());
    container.remove();
  });

  function mount() {
    const counter = loadCounter();
    root = createRoot(container);
    act(() => root.render(<App />));
    commit(container);
    return counter;
  }

  it("counts every component on mount with the mount cause", () => {
    const report = mount().report();

    expect(report.commits).toBe(1);
    expect(causes(report)).toMatchObject({
      "App mount": 1,
      "Counter mount": 1,
      "Leaf mount": 2,
      "Label mount": 1,
      "ThemeReader mount": 1,
    });
  });

  it("names the state slot and skips memoised children whose props held", () => {
    const counter = mount();
    counter.reset();

    act(() => setCount(1));
    commit(container);
    const report = counter.report();

    expect(report.commits).toBe(1);
    expect(causes(report)).toEqual({
      "Counter state#0": 1,
      "Leaf parent-same-props<-Counter": 1,
    });
  });

  it("lists the changed props of a memoised child", () => {
    const counter = mount();
    act(() => setCount(1));
    counter.reset();

    act(() => setCount(2));
    commit(container);

    expect(causes(counter.report())).toEqual({
      "Counter state#0": 1,
      "Leaf parent-same-props<-Counter": 1,
      "Label props:value": 1,
    });
  });

  it("attributes a render under a bailed-out parent to context", () => {
    const counter = mount();
    counter.reset();

    act(() => setTheme("dark"));
    commit(container);
    const byCause = causes(counter.report());

    expect(byCause["ThemeReader context"]).toBe(1);
    expect(byCause["ThemeBox context"]).toBeUndefined();
    expect(byCause["Label props:value"]).toBeUndefined();
  });

  it("records nothing while stopped and starts again from zero after reset", () => {
    const counter = mount();
    counter.stop();
    act(() => setCount(5));
    commit(container);
    expect(counter.report().commits).toBe(1);

    counter.reset();
    counter.start();
    act(() => setCount(6));
    commit(container);
    expect(counter.report().commits).toBe(1);
    expect(counter.report().totalRenders).toBeGreaterThan(0);
  });

  it("keeps calling an existing devtools hook", () => {
    const previous = vi.fn();
    testWindow.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
      onCommitFiberRoot: previous,
    };
    loadCounter();
    root = createRoot(container);
    act(() => root.render(<App />));
    commit(container);

    expect(previous).toHaveBeenCalledTimes(1);
    expect(testWindow.__renderCounter!.report().commits).toBe(1);
  });
});

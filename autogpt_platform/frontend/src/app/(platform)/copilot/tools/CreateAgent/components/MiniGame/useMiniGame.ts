import { useEffect, useRef, useState } from "react";

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */

const CANVAS_HEIGHT = 150;
const CELL_SIZE = 10;
const TICK_MS = 120;
const STORAGE_KEY = "copilot-minigame-highscore";

const COLOR_BG = "#E8EAF6";
const COLOR_SNAKE = "#263238";
const COLOR_SNAKE_HEAD = "#1a1a2e";
const COLOR_FOOD = "#a855f7";
const COLOR_GRID = "rgba(0,0,0,0.04)";

/* ------------------------------------------------------------------ */
/*  Types                                                             */
/* ------------------------------------------------------------------ */

interface Point {
  x: number;
  y: number;
}

type Direction = "up" | "down" | "left" | "right";

interface GameState {
  snake: Point[];
  food: Point;
  dir: Direction;
  nextDir: Direction;
  score: number;
  highScore: number;
  running: boolean;
  over: boolean;
  cols: number;
  rows: number;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

function readHighScore(): number {
  try {
    return parseInt(localStorage.getItem(STORAGE_KEY) || "0", 10) || 0;
  } catch {
    return 0;
  }
}

function writeHighScore(score: number) {
  try {
    localStorage.setItem(STORAGE_KEY, String(score));
  } catch {
    /* noop */
  }
}

function spawnFood(cols: number, rows: number, snake: Point[]): Point | null {
  const occupied = new Set(snake.map((p) => `${p.x},${p.y}`));
  const free: Point[] = [];
  for (let x = 0; x < cols; x++) {
    for (let y = 0; y < rows; y++) {
      if (!occupied.has(`${x},${y}`)) free.push({ x, y });
    }
  }
  if (free.length === 0) return null;
  return free[Math.floor(Math.random() * free.length)];
}

function makeState(cols: number, rows: number): GameState {
  const cx = Math.floor(cols / 2);
  const cy = Math.floor(rows / 2);
  const snake: Point[] = [];
  for (let i = 0; i < 40; i++) {
    snake.push({ x: cx - i, y: cy });
  }
  return {
    snake,
    food: spawnFood(cols, rows, snake)!,
    dir: "right",
    nextDir: "right",
    score: 0,
    highScore: readHighScore(),
    running: false,
    over: false,
    cols,
    rows,
  };
}

const OPPOSITE: Record<Direction, Direction> = {
  up: "down",
  down: "up",
  left: "right",
  right: "left",
};

function tick(s: GameState) {
  if (!s.running || s.over) return;

  // Apply queued direction (prevent 180-degree turns)
  if (s.nextDir !== OPPOSITE[s.dir]) {
    s.dir = s.nextDir;
  }

  const head = s.snake[0];
  let nx = head.x;
  let ny = head.y;

  switch (s.dir) {
    case "up":
      ny--;
      break;
    case "down":
      ny++;
      break;
    case "left":
      nx--;
      break;
    case "right":
      nx++;
      break;
  }

  // Wrap around walls (Nokia style)
  if (nx < 0) nx = s.cols - 1;
  if (nx >= s.cols) nx = 0;
  if (ny < 0) ny = s.rows - 1;
  if (ny >= s.rows) ny = 0;

  // Self-collision
  for (const seg of s.snake) {
    if (seg.x === nx && seg.y === ny) {
      s.over = true;
      s.running = false;
      if (s.score > s.highScore) {
        s.highScore = s.score;
        writeHighScore(s.score);
      }
      return;
    }
  }

  s.snake.unshift({ x: nx, y: ny });

  // Eat food
  if (nx === s.food.x && ny === s.food.y) {
    s.score++;
    const newFood = spawnFood(s.cols, s.rows, s.snake);
    if (newFood) {
      s.food = newFood;
    } else {
      // Grid full — player wins
      s.over = true;
      s.running = false;
      if (s.score > s.highScore) {
        s.highScore = s.score;
        writeHighScore(s.score);
      }
    }
  } else {
    s.snake.pop();
  }
}

/* ------------------------------------------------------------------ */
/*  Drawing                                                           */
/* ------------------------------------------------------------------ */

function draw(
  ctx: CanvasRenderingContext2D,
  s: GameState,
  w: number,
  h: number,
) {
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(0, 0, w, h);

  const offsetX = Math.floor((w - s.cols * CELL_SIZE) / 2);
  const offsetY = Math.floor((h - s.rows * CELL_SIZE) / 2);

  // Grid
  ctx.strokeStyle = COLOR_GRID;
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= s.cols; x++) {
    ctx.beginPath();
    ctx.moveTo(offsetX + x * CELL_SIZE, offsetY);
    ctx.lineTo(offsetX + x * CELL_SIZE, offsetY + s.rows * CELL_SIZE);
    ctx.stroke();
  }
  for (let y = 0; y <= s.rows; y++) {
    ctx.beginPath();
    ctx.moveTo(offsetX, offsetY + y * CELL_SIZE);
    ctx.lineTo(offsetX + s.cols * CELL_SIZE, offsetY + y * CELL_SIZE);
    ctx.stroke();
  }

  // Border
  ctx.strokeStyle = "rgba(0,0,0,0.15)";
  ctx.lineWidth = 1;
  ctx.strokeRect(offsetX, offsetY, s.cols * CELL_SIZE, s.rows * CELL_SIZE);

  // Snake body
  for (let i = 1; i < s.snake.length; i++) {
    const seg = s.snake[i];
    const alpha = 0.9 - (i / s.snake.length) * 0.4;
    ctx.fillStyle = COLOR_SNAKE;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.roundRect(
      offsetX + seg.x * CELL_SIZE + 1,
      offsetY + seg.y * CELL_SIZE + 1,
      CELL_SIZE - 2,
      CELL_SIZE - 2,
      2,
    );
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Snake head
  const head = s.snake[0];
  ctx.fillStyle = COLOR_SNAKE_HEAD;
  ctx.beginPath();
  ctx.roundRect(
    offsetX + head.x * CELL_SIZE + 0.5,
    offsetY + head.y * CELL_SIZE + 0.5,
    CELL_SIZE - 1,
    CELL_SIZE - 1,
    2,
  );
  ctx.fill();

  // Food
  ctx.fillStyle = COLOR_FOOD;
  ctx.beginPath();
  ctx.arc(
    offsetX + s.food.x * CELL_SIZE + CELL_SIZE / 2,
    offsetY + s.food.y * CELL_SIZE + CELL_SIZE / 2,
    CELL_SIZE / 2 - 1,
    0,
    Math.PI * 2,
  );
  ctx.fill();

  // Score HUD
  ctx.save();
  ctx.fillStyle = COLOR_SNAKE;
  ctx.globalAlpha = 0.5;
  ctx.font = "bold 11px monospace";
  ctx.textAlign = "right";
  ctx.fillText(`Score: ${s.score}`, w - 12, 20);
  ctx.fillText(`Best: ${s.highScore}`, w - 12, 34);
  ctx.restore();
}

/* ------------------------------------------------------------------ */
/*  Hook                                                              */
/* ------------------------------------------------------------------ */

export function useMiniGame() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<GameState | null>(null);
  const rafRef = useRef(0);
  const tickTimerRef = useRef(0);
  const startedRef = useRef(false);
  const [activeMode, setActiveMode] = useState<"idle" | "run" | "over">("idle");
  const [showOverlay, setShowOverlay] = useState(true);
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const container = canvas.parentElement;
    if (container) {
      canvas.width = container.clientWidth;
      canvas.height = CANVAS_HEIGHT;
    }

    const cols = Math.floor(canvas.width / CELL_SIZE);
    const rows = Math.floor(CANVAS_HEIGHT / CELL_SIZE);
    stateRef.current = makeState(cols, rows);

    let prevPhase = "";

    function onKeyDown(e: KeyboardEvent) {
      const s = stateRef.current;
      if (!s || s.over) return;

      let handled = true;
      switch (e.code) {
        case "KeyW":
        case "ArrowUp":
          s.nextDir = "up";
          break;
        case "KeyS":
        case "ArrowDown":
          s.nextDir = "down";
          break;
        case "KeyA":
        case "ArrowLeft":
          s.nextDir = "left";
          break;
        case "KeyD":
        case "ArrowRight":
          s.nextDir = "right";
          break;
        default:
          handled = false;
      }
      if (handled) e.preventDefault();
    }

    function onClick() {
      canvas?.focus();
    }

    // Game tick (fixed interval)
    tickTimerRef.current = window.setInterval(() => {
      const s = stateRef.current;
      if (!s) return;
      tick(s);
    }, TICK_MS);

    // Render loop
    function loop() {
      const s = stateRef.current;
      if (!canvas || !s) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      draw(ctx, s, canvas.width, canvas.height);

      let phase: string;
      if (s.over) phase = "over";
      else if (!startedRef.current) phase = "idle";
      else phase = "running";

      if (phase !== prevPhase) {
        prevPhase = phase;
        switch (phase) {
          case "idle":
            setActiveMode("idle");
            setShowOverlay(true);
            break;
          case "running":
            setActiveMode("run");
            setShowOverlay(false);
            break;
          case "over":
            setActiveMode("over");
            setScore(s.score);
            setHighScore(s.highScore);
            setShowOverlay(true);
            break;
        }
      }

      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);

    canvas.addEventListener("click", onClick);
    canvas.addEventListener("keydown", onKeyDown);

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        canvas.width = entry.contentRect.width;
        canvas.height = CANVAS_HEIGHT;
        if (stateRef.current) {
          const s = stateRef.current;
          const newCols = Math.floor(canvas.width / CELL_SIZE);
          const newRows = Math.floor(CANVAS_HEIGHT / CELL_SIZE);
          s.cols = newCols;
          s.rows = newRows;

          // Wrap snake segments that are now outside the smaller grid
          // and deduplicate so overlapping segments don't corrupt state
          const seen = new Set<string>();
          const deduped: Point[] = [];
          for (const seg of s.snake) {
            seg.x = ((seg.x % newCols) + newCols) % newCols;
            seg.y = ((seg.y % newRows) + newRows) % newRows;
            const key = `${seg.x},${seg.y}`;
            if (!seen.has(key)) {
              seen.add(key);
              deduped.push(seg);
            }
          }
          s.snake = deduped;

          // Respawn food if it landed outside the new bounds
          if (s.food.x >= newCols || s.food.y >= newRows) {
            s.food = spawnFood(newCols, newRows, s.snake) ?? {
              x: s.food.x % newCols,
              y: s.food.y % newRows,
            };
          }
        }
      }
    });
    if (container) observer.observe(container);

    return () => {
      cancelAnimationFrame(rafRef.current);
      clearInterval(tickTimerRef.current);
      canvas.removeEventListener("click", onClick);
      canvas.removeEventListener("keydown", onKeyDown);
      observer.disconnect();
    };
  }, []);

  function onContinue() {
    const s = stateRef.current;
    const canvas = canvasRef.current;
    if (!s || !canvas) return;

    if (s.over) {
      const hs = s.highScore;
      const cols = Math.floor(canvas.width / CELL_SIZE);
      const rows = Math.floor(CANVAS_HEIGHT / CELL_SIZE);
      stateRef.current = makeState(cols, rows);
      stateRef.current.highScore = hs;
      stateRef.current.running = true;
      startedRef.current = true;
    } else if (!s.running) {
      s.running = true;
      startedRef.current = true;
    }

    setShowOverlay(false);
    canvas.focus();
  }

  return { canvasRef, activeMode, showOverlay, score, highScore, onContinue };
}

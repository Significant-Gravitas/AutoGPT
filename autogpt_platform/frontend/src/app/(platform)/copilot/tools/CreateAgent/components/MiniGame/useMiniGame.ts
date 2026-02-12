import { useEffect, useRef } from "react";

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */

const CANVAS_HEIGHT = 150;
const GRAVITY = 0.55;
const JUMP_FORCE = -9.5;
const BASE_SPEED = 3;
const SPEED_INCREMENT = 0.0008;
const SPAWN_MIN = 70;
const SPAWN_MAX = 130;
const CHAR_SIZE = 18;
const CHAR_X = 50;
const GROUND_PAD = 20;
const STORAGE_KEY = "copilot-minigame-highscore";

// Colors
const COLOR_BG = "#E8EAF6";
const COLOR_CHAR = "#263238";
const COLOR_BOSS = "#F50057";

// Boss
const BOSS_SIZE = 36;
const BOSS_ENTER_SPEED = 2;
const BOSS_LEAVE_SPEED = 3;
const BOSS_SHOOT_COOLDOWN = 90;
const BOSS_SHOTS_TO_EVADE = 5;
const BOSS_INTERVAL = 20; // every N score
const PROJ_SPEED = 4.5;
const PROJ_SIZE = 12;

/* ------------------------------------------------------------------ */
/*  Types                                                             */
/* ------------------------------------------------------------------ */

interface Obstacle {
  x: number;
  width: number;
  height: number;
  scored: boolean;
}

interface Projectile {
  x: number;
  y: number;
  speed: number;
  evaded: boolean;
  type: "low" | "high";
}

interface BossState {
  phase: "inactive" | "entering" | "fighting" | "leaving";
  x: number;
  targetX: number;
  shotsEvaded: number;
  cooldown: number;
  projectiles: Projectile[];
  bob: number;
}

interface GameState {
  charY: number;
  vy: number;
  obstacles: Obstacle[];
  score: number;
  highScore: number;
  speed: number;
  frame: number;
  nextSpawn: number;
  running: boolean;
  over: boolean;
  groundY: number;
  boss: BossState;
  bossThreshold: number;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

function randInt(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

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

function makeBoss(): BossState {
  return {
    phase: "inactive",
    x: 0,
    targetX: 0,
    shotsEvaded: 0,
    cooldown: 0,
    projectiles: [],
    bob: 0,
  };
}

function makeState(groundY: number): GameState {
  return {
    charY: groundY - CHAR_SIZE,
    vy: 0,
    obstacles: [],
    score: 0,
    highScore: readHighScore(),
    speed: BASE_SPEED,
    frame: 0,
    nextSpawn: randInt(SPAWN_MIN, SPAWN_MAX),
    running: false,
    over: false,
    groundY,
    boss: makeBoss(),
    bossThreshold: BOSS_INTERVAL,
  };
}

function gameOver(s: GameState) {
  s.running = false;
  s.over = true;
  if (s.score > s.highScore) {
    s.highScore = s.score;
    writeHighScore(s.score);
  }
}

/* ------------------------------------------------------------------ */
/*  Projectile collision — shared between fighting & leaving phases   */
/* ------------------------------------------------------------------ */

/** Returns true if the player died. */
function tickProjectiles(s: GameState): boolean {
  const boss = s.boss;

  for (const p of boss.projectiles) {
    p.x -= p.speed;

    if (!p.evaded && p.x + PROJ_SIZE < CHAR_X) {
      p.evaded = true;
      boss.shotsEvaded++;
    }

    // Collision
    if (
      !p.evaded &&
      CHAR_X + CHAR_SIZE > p.x &&
      CHAR_X < p.x + PROJ_SIZE &&
      s.charY + CHAR_SIZE > p.y &&
      s.charY < p.y + PROJ_SIZE
    ) {
      gameOver(s);
      return true;
    }
  }

  boss.projectiles = boss.projectiles.filter((p) => p.x + PROJ_SIZE > -20);
  return false;
}

/* ------------------------------------------------------------------ */
/*  Update                                                            */
/* ------------------------------------------------------------------ */

function update(s: GameState, canvasWidth: number) {
  if (!s.running) return;

  s.frame++;

  // Speed only ramps during regular play
  if (s.boss.phase === "inactive") {
    s.speed = BASE_SPEED + s.frame * SPEED_INCREMENT;
  }

  // ---- Character physics (always active) ---- //
  s.vy += GRAVITY;
  s.charY += s.vy;
  if (s.charY + CHAR_SIZE >= s.groundY) {
    s.charY = s.groundY - CHAR_SIZE;
    s.vy = 0;
  }

  // ---- Trigger boss ---- //
  if (s.boss.phase === "inactive" && s.score >= s.bossThreshold) {
    s.boss.phase = "entering";
    s.boss.x = canvasWidth + 10;
    s.boss.targetX = canvasWidth - BOSS_SIZE - 40;
    s.boss.shotsEvaded = 0;
    s.boss.cooldown = BOSS_SHOOT_COOLDOWN;
    s.boss.projectiles = [];
    s.obstacles = [];
  }

  // ---- Boss: entering ---- //
  if (s.boss.phase === "entering") {
    s.boss.bob = Math.sin(s.frame * 0.05) * 3;
    s.boss.x -= BOSS_ENTER_SPEED;
    if (s.boss.x <= s.boss.targetX) {
      s.boss.x = s.boss.targetX;
      s.boss.phase = "fighting";
    }
    return; // no obstacles while entering
  }

  // ---- Boss: fighting ---- //
  if (s.boss.phase === "fighting") {
    s.boss.bob = Math.sin(s.frame * 0.05) * 3;

    // Shoot
    s.boss.cooldown--;
    if (s.boss.cooldown <= 0) {
      const isLow = Math.random() < 0.5;
      s.boss.projectiles.push({
        x: s.boss.x - PROJ_SIZE,
        y: isLow ? s.groundY - 14 : s.groundY - 70,
        speed: PROJ_SPEED,
        evaded: false,
        type: isLow ? "low" : "high",
      });
      s.boss.cooldown = BOSS_SHOOT_COOLDOWN;
    }

    if (tickProjectiles(s)) return;

    // Boss defeated?
    if (s.boss.shotsEvaded >= BOSS_SHOTS_TO_EVADE) {
      s.boss.phase = "leaving";
      s.score += 5; // bonus
      s.bossThreshold = s.score + BOSS_INTERVAL;
    }
    return;
  }

  // ---- Boss: leaving ---- //
  if (s.boss.phase === "leaving") {
    s.boss.bob = Math.sin(s.frame * 0.05) * 3;
    s.boss.x += BOSS_LEAVE_SPEED;

    // Still check in-flight projectiles
    if (tickProjectiles(s)) return;

    if (s.boss.x > canvasWidth + 50) {
      s.boss = makeBoss();
      s.nextSpawn = s.frame + randInt(SPAWN_MIN / 2, SPAWN_MAX / 2);
    }
    return;
  }

  // ---- Regular obstacle play ---- //
  if (s.frame >= s.nextSpawn) {
    s.obstacles.push({
      x: canvasWidth + 10,
      width: randInt(10, 16),
      height: randInt(20, 48),
      scored: false,
    });
    s.nextSpawn = s.frame + randInt(SPAWN_MIN, SPAWN_MAX);
  }

  for (const o of s.obstacles) {
    o.x -= s.speed;
    if (!o.scored && o.x + o.width < CHAR_X) {
      o.scored = true;
      s.score++;
    }
  }

  s.obstacles = s.obstacles.filter((o) => o.x + o.width > -20);

  for (const o of s.obstacles) {
    const oY = s.groundY - o.height;
    if (
      CHAR_X + CHAR_SIZE > o.x &&
      CHAR_X < o.x + o.width &&
      s.charY + CHAR_SIZE > oY
    ) {
      gameOver(s);
      return;
    }
  }
}

/* ------------------------------------------------------------------ */
/*  Drawing                                                           */
/* ------------------------------------------------------------------ */

function drawBoss(ctx: CanvasRenderingContext2D, s: GameState, bg: string) {
  const bx = s.boss.x;
  const by = s.groundY - BOSS_SIZE + s.boss.bob;

  // Body
  ctx.save();
  ctx.fillStyle = COLOR_BOSS;
  ctx.globalAlpha = 0.9;
  ctx.beginPath();
  ctx.roundRect(bx, by, BOSS_SIZE, BOSS_SIZE, 4);
  ctx.fill();
  ctx.restore();

  // Eyes
  ctx.save();
  ctx.fillStyle = bg;
  const eyeY = by + 13;
  ctx.beginPath();
  ctx.arc(bx + 10, eyeY, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(bx + 26, eyeY, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Angry eyebrows
  ctx.save();
  ctx.strokeStyle = bg;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(bx + 5, eyeY - 7);
  ctx.lineTo(bx + 14, eyeY - 4);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(bx + 31, eyeY - 7);
  ctx.lineTo(bx + 22, eyeY - 4);
  ctx.stroke();
  ctx.restore();

  // Zigzag mouth
  ctx.save();
  ctx.strokeStyle = bg;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(bx + 10, by + 27);
  ctx.lineTo(bx + 14, by + 24);
  ctx.lineTo(bx + 18, by + 27);
  ctx.lineTo(bx + 22, by + 24);
  ctx.lineTo(bx + 26, by + 27);
  ctx.stroke();
  ctx.restore();
}

function drawProjectiles(ctx: CanvasRenderingContext2D, boss: BossState) {
  ctx.save();
  ctx.fillStyle = COLOR_BOSS;
  ctx.globalAlpha = 0.8;
  for (const p of boss.projectiles) {
    if (p.evaded) continue;
    ctx.beginPath();
    ctx.arc(
      p.x + PROJ_SIZE / 2,
      p.y + PROJ_SIZE / 2,
      PROJ_SIZE / 2,
      0,
      Math.PI * 2,
    );
    ctx.fill();
  }
  ctx.restore();
}

function draw(
  ctx: CanvasRenderingContext2D,
  s: GameState,
  w: number,
  h: number,
  fg: string,
  started: boolean,
) {
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(0, 0, w, h);

  // Ground
  ctx.save();
  ctx.strokeStyle = fg;
  ctx.globalAlpha = 0.15;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, s.groundY);
  ctx.lineTo(w, s.groundY);
  ctx.stroke();
  ctx.restore();

  // Character
  ctx.save();
  ctx.fillStyle = COLOR_CHAR;
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  ctx.roundRect(CHAR_X, s.charY, CHAR_SIZE, CHAR_SIZE, 3);
  ctx.fill();
  ctx.restore();

  // Eyes
  ctx.save();
  ctx.fillStyle = COLOR_BG;
  ctx.beginPath();
  ctx.arc(CHAR_X + 6, s.charY + 7, 2.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(CHAR_X + 12, s.charY + 7, 2.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Obstacles
  ctx.save();
  ctx.fillStyle = fg;
  ctx.globalAlpha = 0.55;
  for (const o of s.obstacles) {
    ctx.fillRect(o.x, s.groundY - o.height, o.width, o.height);
  }
  ctx.restore();

  // Boss + projectiles
  if (s.boss.phase !== "inactive") {
    drawBoss(ctx, s, COLOR_BG);
    drawProjectiles(ctx, s.boss);
  }

  // Score HUD
  ctx.save();
  ctx.fillStyle = fg;
  ctx.globalAlpha = 0.5;
  ctx.font = "bold 11px monospace";
  ctx.textAlign = "right";
  ctx.fillText(`Score: ${s.score}`, w - 12, 20);
  ctx.fillText(`Best: ${s.highScore}`, w - 12, 34);
  if (s.boss.phase === "fighting") {
    ctx.fillText(
      `Evade: ${s.boss.shotsEvaded}/${BOSS_SHOTS_TO_EVADE}`,
      w - 12,
      48,
    );
  }
  ctx.restore();

  // Prompts
  if (!started && !s.running && !s.over) {
    ctx.save();
    ctx.fillStyle = fg;
    ctx.globalAlpha = 0.5;
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Click or press Space to play while you wait", w / 2, h / 2);
    ctx.restore();
  }

  if (s.over) {
    ctx.save();
    ctx.fillStyle = fg;
    ctx.globalAlpha = 0.7;
    ctx.font = "bold 13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Game Over", w / 2, h / 2 - 8);
    ctx.font = "11px sans-serif";
    ctx.fillText("Click or Space to restart", w / 2, h / 2 + 10);
    ctx.restore();
  }
}

/* ------------------------------------------------------------------ */
/*  Hook                                                              */
/* ------------------------------------------------------------------ */

export function useMiniGame() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<GameState | null>(null);
  const rafRef = useRef(0);
  const startedRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const container = canvas.parentElement;
    if (container) {
      canvas.width = container.clientWidth;
      canvas.height = CANVAS_HEIGHT;
    }

    const groundY = canvas.height - GROUND_PAD;
    stateRef.current = makeState(groundY);

    const style = getComputedStyle(canvas);
    let fg = style.color || "#71717a";

    // -------------------------------------------------------------- //
    //  Jump                                                           //
    // -------------------------------------------------------------- //
    function jump() {
      const s = stateRef.current;
      if (!s) return;

      if (s.over) {
        const hs = s.highScore;
        const gy = s.groundY;
        stateRef.current = makeState(gy);
        stateRef.current.highScore = hs;
        stateRef.current.running = true;
        startedRef.current = true;
        return;
      }

      if (!s.running) {
        s.running = true;
        startedRef.current = true;
        return;
      }

      // Only jump when on the ground
      if (s.charY + CHAR_SIZE >= s.groundY) {
        s.vy = JUMP_FORCE;
      }
    }

    function onKey(e: KeyboardEvent) {
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        jump();
      }
    }

    function onClick() {
      canvas?.focus();
      jump();
    }

    // -------------------------------------------------------------- //
    //  Loop                                                           //
    // -------------------------------------------------------------- //
    function loop() {
      const s = stateRef.current;
      if (!canvas || !s) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      update(s, canvas.width);
      draw(ctx, s, canvas.width, canvas.height, fg, startedRef.current);
      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);

    canvas.addEventListener("click", onClick);
    canvas.addEventListener("keydown", onKey);

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        canvas.width = entry.contentRect.width;
        canvas.height = CANVAS_HEIGHT;
        if (stateRef.current) {
          stateRef.current.groundY = canvas.height - GROUND_PAD;
        }
        const cs = getComputedStyle(canvas);
        fg = cs.color || fg;
      }
    });
    if (container) observer.observe(container);

    return () => {
      cancelAnimationFrame(rafRef.current);
      canvas.removeEventListener("click", onClick);
      canvas.removeEventListener("keydown", onKey);
      observer.disconnect();
    };
  }, []);

  return { canvasRef };
}

import { useEffect, useRef, useState } from "react";
import runSheet from "./assets/run.png";
import idleSheet from "./assets/idle.png";
import attackSheet from "./assets/attack.png";
import tree1Sheet from "./assets/tree-1.png";
import tree2Sheet from "./assets/tree-2.png";
import tree3Sheet from "./assets/tree-3.png";
import archerIdleSheet from "./assets/archer-idle.png";
import archerAttackSheet from "./assets/archer-attack.png";
import guardSheet from "./assets/guard.png";

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
const CHAR_SPRITE_SIZE = 67;
const CHAR_X = 50;
const GROUND_PAD = 20;
const STORAGE_KEY = "copilot-minigame-highscore";

// Character sprite sheets (each frame is 192x192)
const SPRITE_FRAME_SIZE = 192;
const RUN_FRAMES = 6;
const IDLE_FRAMES = 8;
const ATTACK_FRAMES = 4;
const ANIM_SPEED = 8;
const ATTACK_ANIM_SPEED = 6;
const ATTACK_RANGE = 40;
const ATTACK_HIT_FRAME = 2;
const GUARD_FRAMES = 6;
const GUARD_ANIM_SPEED = 8;

// Tree sprite sheets: 8 frames each, 192px wide per frame
const TREE_FRAMES = 8;
const TREE_ANIM_SPEED = 10;
const TREE_CONFIGS = [
  { frameW: 192, frameH: 256, renderW: 40, renderH: 61, hitW: 16, hitH: 50 },
  { frameW: 192, frameH: 192, renderW: 38, renderH: 52, hitW: 16, hitH: 40 },
  { frameW: 192, frameH: 192, renderW: 32, renderH: 40, hitW: 14, hitH: 30 },
] as const;

// Colors
const COLOR_BG = "#E8EAF6";
const COLOR_CHAR = "#263238";

// Boss
const BOSS_SIZE = 36;
const BOSS_SPRITE_SIZE = 70;
const BOSS_ENTER_SPEED = 2;
const BOSS_HP = 1;
const MOVE_SPEED = 3;
const BOSS_CHASE_SPEED = 2.2;
const BOSS_RETREAT_SPEED = 2;
const BOSS_ATTACK_RANGE = 50;
const BOSS_IDLE_TIME = 166;
const BOSS_RETREAT_TIME = 166;

// Archer sprite sheets
const ARCHER_IDLE_FRAMES = 6;
const ARCHER_ATTACK_FRAMES = 4;
const ARCHER_FRAME_SIZE = 192;
const ARCHER_ANIM_SPEED = 8;
const ARCHER_ATTACK_ANIM_SPEED = 6;
const ARCHER_ATTACK_HIT_FRAME = 2;

// Death animation
const DEATH_PARTICLE_COUNT = 15;
const DEATH_ANIM_DURATION = 40;

// Attack effect
const ATTACK_EFFECT_COUNT = 8;
const ATTACK_EFFECT_DURATION = 15;

/* ------------------------------------------------------------------ */
/*  Types                                                             */
/* ------------------------------------------------------------------ */

interface Obstacle {
  x: number;
  width: number;
  height: number;
  scored: boolean;
  treeType: 0 | 1 | 2;
}

interface BossState {
  phase: "inactive" | "entering" | "fighting";
  x: number;
  y: number;
  vy: number;
  targetX: number;
  hp: number;
  action: "idle" | "chase" | "retreat" | "attack";
  actionTimer: number;
  attackFrame: number;
  attackHit: boolean;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
}

interface DeathAnim {
  particles: Particle[];
  type: "boss" | "player";
  timer: number;
}

interface GameState {
  charX: number;
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
  bossesDefeated: number;
  paused: boolean;
  nextTreeType: 0 | 1 | 2;
  attacking: boolean;
  attackFrame: number;
  attackHit: boolean;
  guarding: boolean;
  guardFrame: number;
  deathAnim: DeathAnim | null;
  attackEffects: Particle[];
}

interface KeyState {
  left: boolean;
  right: boolean;
}

interface Sprites {
  run: HTMLImageElement;
  idle: HTMLImageElement;
  attack: HTMLImageElement;
  guard: HTMLImageElement;
  trees: HTMLImageElement[];
  archerIdle: HTMLImageElement;
  archerAttack: HTMLImageElement;
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

function makeBoss(groundY: number): BossState {
  return {
    phase: "inactive",
    x: 0,
    y: groundY - BOSS_SIZE,
    vy: 0,
    targetX: 0,
    hp: BOSS_HP,
    action: "idle",
    actionTimer: BOSS_IDLE_TIME,
    attackFrame: 0,
    attackHit: false,
  };
}

function makeState(groundY: number): GameState {
  return {
    charX: CHAR_X,
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
    boss: makeBoss(groundY),
    bossThreshold: 10,
    bossesDefeated: 0,
    paused: false,
    nextTreeType: 0,
    attacking: false,
    attackFrame: 0,
    attackHit: false,
    guarding: false,
    guardFrame: 0,
    deathAnim: null,
    attackEffects: [],
  };
}

function spawnParticles(x: number, y: number): Particle[] {
  const particles: Particle[] = [];
  for (let i = 0; i < DEATH_PARTICLE_COUNT; i++) {
    const angle = Math.random() * Math.PI * 2;
    const speed = 1 + Math.random() * 3;
    particles.push({
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - 2,
      life: DEATH_ANIM_DURATION,
    });
  }
  return particles;
}

function startPlayerDeath(s: GameState) {
  s.deathAnim = {
    particles: spawnParticles(s.charX + CHAR_SIZE / 2, s.charY + CHAR_SIZE / 2),
    type: "player",
    timer: DEATH_ANIM_DURATION,
  };
}

function startBossDeath(s: GameState) {
  s.deathAnim = {
    particles: spawnParticles(
      s.boss.x + BOSS_SIZE / 2,
      s.boss.y + BOSS_SIZE / 2,
    ),
    type: "boss",
    timer: DEATH_ANIM_DURATION,
  };
}

/* ------------------------------------------------------------------ */
/*  Update                                                            */
/* ------------------------------------------------------------------ */

function update(s: GameState, canvasWidth: number, keys: KeyState) {
  if (!s.running || s.paused) return;

  s.frame++;

  // ---- Attack effects ---- //
  for (const p of s.attackEffects) {
    p.x += p.vx;
    p.y += p.vy;
    p.vy += 0.08;
    p.life--;
  }
  s.attackEffects = s.attackEffects.filter((p) => p.life > 0);

  // ---- Death animation ---- //
  if (s.deathAnim) {
    s.deathAnim.timer--;
    for (const p of s.deathAnim.particles) {
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.1;
      p.life--;
    }
    if (s.deathAnim.timer <= 0) {
      if (s.deathAnim.type === "player") {
        s.deathAnim = null;
        s.running = false;
        s.over = true;
        if (s.score > s.highScore) {
          s.highScore = s.score;
          writeHighScore(s.score);
        }
      } else {
        s.deathAnim = null;
        s.score += 10;
        s.bossesDefeated++;
        if (s.bossesDefeated === 1) {
          s.bossThreshold = s.score + 15;
        } else {
          s.bossThreshold = s.score + 20;
        }
        s.paused = true;
      }
    }
    return;
  }

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

  // ---- Attack animation ---- //
  if (s.attacking) {
    s.attackFrame++;

    if (
      !s.attackHit &&
      Math.floor(s.attackFrame / ATTACK_ANIM_SPEED) === ATTACK_HIT_FRAME &&
      s.boss.phase === "fighting" &&
      s.charX + CHAR_SIZE + ATTACK_RANGE >= s.boss.x
    ) {
      s.boss.hp--;
      s.attackHit = true;
    }

    if (s.attackFrame >= ATTACK_FRAMES * ATTACK_ANIM_SPEED) {
      s.attacking = false;
      s.attackFrame = 0;
      s.attackHit = false;
    }
  }

  // ---- Guard animation ---- //
  if (s.guarding) {
    s.guardFrame++;
    if (s.guardFrame >= GUARD_FRAMES * GUARD_ANIM_SPEED) {
      s.guardFrame = GUARD_FRAMES * GUARD_ANIM_SPEED - 1;
    }
  }

  // ---- Horizontal movement during boss fight ---- //
  if (s.boss.phase !== "inactive") {
    if (keys.left) {
      s.charX = Math.max(10, s.charX - MOVE_SPEED);
    }
    if (keys.right) {
      s.charX = Math.min(canvasWidth - CHAR_SIZE - 10, s.charX + MOVE_SPEED);
    }
  } else {
    s.charX = CHAR_X;
  }

  // ---- Trigger boss ---- //
  const isOnGround = s.charY + CHAR_SIZE >= s.groundY;
  if (
    s.boss.phase === "inactive" &&
    s.score >= s.bossThreshold &&
    s.obstacles.length === 0 &&
    isOnGround
  ) {
    s.boss.phase = "entering";
    s.boss.x = canvasWidth + 10;
    s.boss.y = s.groundY - BOSS_SIZE;
    s.boss.vy = 0;
    s.boss.targetX = canvasWidth - BOSS_SIZE - 40;
    s.boss.hp = BOSS_HP;
    s.boss.action = "idle";
    s.boss.actionTimer = BOSS_IDLE_TIME;
    s.boss.attackFrame = 0;
    s.boss.attackHit = false;

    if (s.bossesDefeated === 0) {
      s.paused = true;
    }
  }

  // ---- Boss: entering ---- //
  if (s.boss.phase === "entering") {
    s.boss.x -= BOSS_ENTER_SPEED;
    if (s.boss.x <= s.boss.targetX) {
      s.boss.x = s.boss.targetX;
      s.boss.phase = "fighting";
    }
    return;
  }

  // ---- Boss: fighting ---- //
  if (s.boss.phase === "fighting") {
    // Boss physics
    s.boss.vy += GRAVITY;
    s.boss.y += s.boss.vy;
    if (s.boss.y + BOSS_SIZE >= s.groundY) {
      s.boss.y = s.groundY - BOSS_SIZE;
      s.boss.vy = 0;
    }

    // Boss defeated?
    if (s.boss.hp <= 0) {
      startBossDeath(s);
      return;
    }

    // Boss AI
    if (s.boss.action === "attack") {
      s.boss.attackFrame++;
      const hitFrame = Math.floor(
        s.boss.attackFrame / ARCHER_ATTACK_ANIM_SPEED,
      );

      // Spawn yellow attack effect at hit frame
      if (
        s.boss.attackFrame ===
        ARCHER_ATTACK_HIT_FRAME * ARCHER_ATTACK_ANIM_SPEED
      ) {
        const effectX = s.boss.x - 5;
        const effectY = s.boss.y + BOSS_SIZE / 2;
        for (let i = 0; i < ATTACK_EFFECT_COUNT; i++) {
          const angle = Math.PI + (Math.random() - 0.5) * 1.2;
          const speed = 2 + Math.random() * 3;
          s.attackEffects.push({
            x: effectX,
            y: effectY,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed - 1,
            life: ATTACK_EFFECT_DURATION,
          });
        }
      }

      if (!s.boss.attackHit && hitFrame === ARCHER_ATTACK_HIT_FRAME) {
        const dist = s.boss.x - (s.charX + CHAR_SIZE);
        if (dist < BOSS_ATTACK_RANGE && dist > -BOSS_SIZE) {
          s.boss.attackHit = true;
          if (!s.guarding) {
            startPlayerDeath(s);
            return;
          }
        }
      }

      if (
        s.boss.attackFrame >=
        ARCHER_ATTACK_FRAMES * ARCHER_ATTACK_ANIM_SPEED
      ) {
        s.boss.action = "retreat";
        s.boss.actionTimer = BOSS_RETREAT_TIME;
        s.boss.attackFrame = 0;
        s.boss.attackHit = false;
      }
    } else {
      s.boss.actionTimer--;

      if (s.boss.action === "chase") {
        if (s.boss.x > s.charX + CHAR_SIZE) {
          s.boss.x -= BOSS_CHASE_SPEED;
        } else {
          s.boss.x += BOSS_CHASE_SPEED;
        }

        // Occasional jump
        if (s.boss.y + BOSS_SIZE >= s.groundY && Math.random() < 0.008) {
          s.boss.vy = JUMP_FORCE * 0.7;
        }

        // Close enough to attack
        const dist = Math.abs(s.boss.x - (s.charX + CHAR_SIZE));
        if (dist < BOSS_ATTACK_RANGE) {
          s.boss.action = "attack";
          s.boss.attackFrame = 0;
          s.boss.attackHit = false;
        }
      } else if (s.boss.action === "retreat") {
        s.boss.x += BOSS_RETREAT_SPEED;
        if (s.boss.x > canvasWidth - BOSS_SIZE - 10) {
          s.boss.x = canvasWidth - BOSS_SIZE - 10;
        }
      }

      // Timer expired → next action
      if (s.boss.actionTimer <= 0) {
        if (s.boss.action === "idle" || s.boss.action === "retreat") {
          s.boss.action = "chase";
          s.boss.actionTimer = 999;
        } else {
          s.boss.action = "idle";
          s.boss.actionTimer = BOSS_IDLE_TIME;
        }
      }
    }
    return;
  }

  // ---- Regular obstacle play ---- //
  // Stop spawning trees if enough are queued to reach boss threshold
  const unscoredCount = s.obstacles.filter((o) => !o.scored).length;
  if (s.score + unscoredCount < s.bossThreshold && s.frame >= s.nextSpawn) {
    const tt = s.nextTreeType;
    const cfg = TREE_CONFIGS[tt];
    s.obstacles.push({
      x: canvasWidth + 10,
      width: cfg.hitW,
      height: cfg.hitH,
      scored: false,
      treeType: tt,
    });
    s.nextTreeType = Math.floor(Math.random() * 3) as 0 | 1 | 2;
    s.nextSpawn = s.frame + randInt(SPAWN_MIN, SPAWN_MAX);
  }

  for (const o of s.obstacles) {
    o.x -= s.speed;
    if (!o.scored && o.x + o.width < s.charX) {
      o.scored = true;
      s.score++;
    }
  }

  s.obstacles = s.obstacles.filter((o) => o.x + o.width > -20);

  for (const o of s.obstacles) {
    const oY = s.groundY - o.height;
    if (
      s.charX + CHAR_SIZE > o.x &&
      s.charX < o.x + o.width &&
      s.charY + CHAR_SIZE > oY
    ) {
      startPlayerDeath(s);
      return;
    }
  }
}

/* ------------------------------------------------------------------ */
/*  Drawing                                                           */
/* ------------------------------------------------------------------ */

function drawBoss(
  ctx: CanvasRenderingContext2D,
  s: GameState,
  sprites: Sprites,
) {
  const boss = s.boss;
  const isAttacking = boss.action === "attack";
  const sheet = isAttacking ? sprites.archerAttack : sprites.archerIdle;
  const totalFrames = isAttacking ? ARCHER_ATTACK_FRAMES : ARCHER_IDLE_FRAMES;
  const animSpeed = isAttacking ? ARCHER_ATTACK_ANIM_SPEED : ARCHER_ANIM_SPEED;

  let frameIndex: number;
  if (isAttacking) {
    frameIndex = Math.min(
      Math.floor(boss.attackFrame / animSpeed),
      totalFrames - 1,
    );
  } else {
    frameIndex = Math.floor(s.frame / animSpeed) % totalFrames;
  }

  const srcX = frameIndex * ARCHER_FRAME_SIZE;
  const spriteDrawX = boss.x + (BOSS_SIZE - BOSS_SPRITE_SIZE) / 2;
  const spriteDrawY = boss.y + BOSS_SIZE - BOSS_SPRITE_SIZE + 12;

  if (sheet.complete && sheet.naturalWidth > 0) {
    ctx.drawImage(
      sheet,
      srcX,
      0,
      ARCHER_FRAME_SIZE,
      ARCHER_FRAME_SIZE,
      spriteDrawX,
      spriteDrawY,
      BOSS_SPRITE_SIZE,
      BOSS_SPRITE_SIZE,
    );
  } else {
    ctx.save();
    ctx.fillStyle = "#F50057";
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    ctx.roundRect(boss.x, boss.y, BOSS_SIZE, BOSS_SIZE, 4);
    ctx.fill();
    ctx.restore();
  }
}

function drawParticles(ctx: CanvasRenderingContext2D, anim: DeathAnim) {
  ctx.save();
  for (const p of anim.particles) {
    if (p.life <= 0) continue;
    const alpha = p.life / DEATH_ANIM_DURATION;
    const size = 2 + alpha * 3;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = "#a855f7";
    ctx.beginPath();
    ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawAttackEffects(ctx: CanvasRenderingContext2D, effects: Particle[]) {
  ctx.save();
  for (const p of effects) {
    if (p.life <= 0) continue;
    const alpha = p.life / ATTACK_EFFECT_DURATION;
    const size = 1.5 + alpha * 2.5;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = "#facc15";
    ctx.beginPath();
    ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
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
  sprites: Sprites,
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

  // Character sprite (hidden during player death)
  if (!s.deathAnim || s.deathAnim.type !== "player") {
    const isJumping = s.charY + CHAR_SIZE < s.groundY;
    let sheet: HTMLImageElement;
    let totalFrames: number;
    let frameIndex: number;

    if (s.guarding) {
      sheet = sprites.guard;
      totalFrames = GUARD_FRAMES;
      frameIndex = Math.min(
        Math.floor(s.guardFrame / GUARD_ANIM_SPEED),
        totalFrames - 1,
      );
    } else if (s.attacking) {
      sheet = sprites.attack;
      totalFrames = ATTACK_FRAMES;
      frameIndex = Math.min(
        Math.floor(s.attackFrame / ATTACK_ANIM_SPEED),
        totalFrames - 1,
      );
    } else if (isJumping) {
      sheet = sprites.idle;
      totalFrames = IDLE_FRAMES;
      frameIndex = Math.floor(s.frame / ANIM_SPEED) % totalFrames;
    } else {
      sheet = sprites.run;
      totalFrames = RUN_FRAMES;
      frameIndex = Math.floor(s.frame / ANIM_SPEED) % totalFrames;
    }

    const srcX = frameIndex * SPRITE_FRAME_SIZE;
    const drawX = s.charX + (CHAR_SIZE - CHAR_SPRITE_SIZE) / 2;
    const drawY = s.charY + CHAR_SIZE - CHAR_SPRITE_SIZE + 15;

    if (sheet.complete && sheet.naturalWidth > 0) {
      ctx.drawImage(
        sheet,
        srcX,
        0,
        SPRITE_FRAME_SIZE,
        SPRITE_FRAME_SIZE,
        drawX,
        drawY,
        CHAR_SPRITE_SIZE,
        CHAR_SPRITE_SIZE,
      );
    } else {
      ctx.save();
      ctx.fillStyle = COLOR_CHAR;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.roundRect(s.charX, s.charY, CHAR_SIZE, CHAR_SIZE, 3);
      ctx.fill();
      ctx.restore();
    }
  }

  // Tree obstacles
  const treeFrame = Math.floor(s.frame / TREE_ANIM_SPEED) % TREE_FRAMES;
  for (const o of s.obstacles) {
    const cfg = TREE_CONFIGS[o.treeType];
    const treeImg = sprites.trees[o.treeType];
    if (treeImg.complete && treeImg.naturalWidth > 0) {
      const treeSrcX = treeFrame * cfg.frameW;
      const treeDrawX = o.x + (o.width - cfg.renderW) / 2;
      const treeDrawY = s.groundY - cfg.renderH;
      ctx.drawImage(
        treeImg,
        treeSrcX,
        0,
        cfg.frameW,
        cfg.frameH,
        treeDrawX,
        treeDrawY,
        cfg.renderW,
        cfg.renderH,
      );
    } else {
      ctx.save();
      ctx.fillStyle = fg;
      ctx.globalAlpha = 0.55;
      ctx.fillRect(o.x, s.groundY - o.height, o.width, o.height);
      ctx.restore();
    }
  }

  // Boss (hidden during boss death)
  if (
    s.boss.phase !== "inactive" &&
    (!s.deathAnim || s.deathAnim.type !== "boss")
  ) {
    drawBoss(ctx, s, sprites);
  }

  // Attack effects
  if (s.attackEffects.length > 0) {
    drawAttackEffects(ctx, s.attackEffects);
  }

  // Death particles
  if (s.deathAnim) {
    drawParticles(ctx, s.deathAnim);
  }

  // Score HUD
  ctx.save();
  ctx.fillStyle = fg;
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
  const startedRef = useRef(false);
  const keysRef = useRef<KeyState>({ left: false, right: false });
  const [activeMode, setActiveMode] = useState<
    "idle" | "run" | "boss" | "over" | "boss-intro" | "boss-defeated"
  >("idle");
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

    const groundY = canvas.height - GROUND_PAD;
    stateRef.current = makeState(groundY);

    const style = getComputedStyle(canvas);
    let fg = style.color || "#71717a";

    // Load sprite sheets
    const sprites: Sprites = {
      run: new Image(),
      idle: new Image(),
      attack: new Image(),
      guard: new Image(),
      trees: [new Image(), new Image(), new Image()],
      archerIdle: new Image(),
      archerAttack: new Image(),
    };
    sprites.run.src = runSheet.src;
    sprites.idle.src = idleSheet.src;
    sprites.attack.src = attackSheet.src;
    sprites.guard.src = guardSheet.src;
    sprites.trees[0].src = tree1Sheet.src;
    sprites.trees[1].src = tree2Sheet.src;
    sprites.trees[2].src = tree3Sheet.src;
    sprites.archerIdle.src = archerIdleSheet.src;
    sprites.archerAttack.src = archerAttackSheet.src;

    let prevPhase = "";

    // -------------------------------------------------------------- //
    //  Input                                                          //
    // -------------------------------------------------------------- //
    function jump() {
      const s = stateRef.current;
      if (!s || !s.running || s.paused || s.over || s.deathAnim) return;

      if (s.charY + CHAR_SIZE >= s.groundY) {
        s.vy = JUMP_FORCE;
      }
    }

    function attack() {
      const s = stateRef.current;
      if (!s || !s.running || s.attacking || s.guarding || s.deathAnim) return;
      s.attacking = true;
      s.attackFrame = 0;
      s.attackHit = false;
    }

    function guardStart() {
      const s = stateRef.current;
      if (!s || !s.running || s.attacking || s.deathAnim) return;
      if (!s.guarding) {
        s.guarding = true;
        s.guardFrame = 0;
      }
    }

    function guardEnd() {
      const s = stateRef.current;
      if (!s) return;
      s.guarding = false;
      s.guardFrame = 0;
    }

    function onKeyDown(e: KeyboardEvent) {
      if (e.code === "Space" || e.key === " ") {
        e.preventDefault();
        jump();
      }
      if (e.code === "KeyZ") {
        e.preventDefault();
        attack();
      }
      if (e.code === "KeyX") {
        e.preventDefault();
        guardStart();
      }
      if (e.code === "ArrowLeft") {
        e.preventDefault();
        keysRef.current.left = true;
      }
      if (e.code === "ArrowRight") {
        e.preventDefault();
        keysRef.current.right = true;
      }
    }

    function onKeyUp(e: KeyboardEvent) {
      if (e.code === "ArrowLeft") keysRef.current.left = false;
      if (e.code === "ArrowRight") keysRef.current.right = false;
      if (e.code === "KeyX") guardEnd();
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

      update(s, canvas.width, keysRef.current);
      draw(ctx, s, canvas.width, canvas.height, fg, sprites);

      // Update active mode on phase change
      let phase: string;
      if (s.over) phase = "over";
      else if (!startedRef.current) phase = "idle";
      else if (s.paused && s.boss.hp <= 0) phase = "boss-defeated";
      else if (s.paused) phase = "boss-intro";
      else if (s.boss.phase !== "inactive") phase = "boss";
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
          case "boss-intro":
            setActiveMode("boss-intro");
            setShowOverlay(true);
            break;
          case "boss":
            setActiveMode("boss");
            setShowOverlay(false);
            break;
          case "boss-defeated":
            setActiveMode("boss-defeated");
            setShowOverlay(true);
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
    canvas.addEventListener("keyup", onKeyUp);

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
      canvas.removeEventListener("keydown", onKeyDown);
      canvas.removeEventListener("keyup", onKeyUp);
      observer.disconnect();
    };
  }, []);

  function onContinue() {
    const s = stateRef.current;
    if (!s) return;

    if (s.over) {
      // Restart after game over
      const hs = s.highScore;
      const gy = s.groundY;
      stateRef.current = makeState(gy);
      stateRef.current.highScore = hs;
      stateRef.current.running = true;
      startedRef.current = true;
    } else if (!s.running) {
      // Start game from idle
      s.running = true;
      startedRef.current = true;
    } else if (s.boss.hp <= 0) {
      // Boss defeated — reset boss, resume running
      s.boss = makeBoss(s.groundY);
      s.charX = CHAR_X;
      s.nextSpawn = s.frame + randInt(SPAWN_MIN / 2, SPAWN_MAX / 2);
      s.paused = false;
    } else {
      // Boss intro — unpause
      s.paused = false;
    }

    setShowOverlay(false);
    canvasRef.current?.focus();
  }

  return { canvasRef, activeMode, showOverlay, score, highScore, onContinue };
}

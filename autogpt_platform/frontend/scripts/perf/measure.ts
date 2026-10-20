/**
 * Measures one scenario: React renders (via the dev render counter), main
 * thread time, long tasks and heap after GC. Writes one JSON per run to
 * .perf/<scenario>/ and prints a table. Docs:
 * docs/engineering/frontend-perf-measurement.md
 *
 *   NEXT_PUBLIC_RENDER_COUNTER=1 pnpm dev     # in another terminal
 *   pnpm perf --scenario copilot-stream-reply --label before --runs 3
 */
import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import { parseArgs } from "util";
import { chromium, type Browser } from "@playwright/test";
import { signIn } from "./lib/auth";
import { heapAfterGC, openCDP, startWindow, stopWindow } from "./lib/cdp";
import { scenarioOutputDir } from "./lib/paths";
import {
  assertCounterLoaded,
  counter,
  counterReport,
} from "./lib/render-counter";
import {
  formatRun,
  formatSummary,
  runFileName,
  safeLabel,
  writeRun,
  type RunResult,
} from "./lib/report";
import { commonHandlers, installRoutes } from "./lib/routes";
import type { Scenario } from "./lib/scenario";
import { startStreamServer, type StreamServer } from "./lib/stream-server";
import { SCENARIOS, findScenario } from "./scenarios/registry";

const RANKING_LIMIT = 50;

function parseOptions() {
  const { values } = parseArgs({
    options: {
      scenario: { type: "string" },
      label: { type: "string", default: "any" },
      runs: { type: "string", default: "3" },
      warmup: { type: "string", default: "1" },
      "base-url": { type: "string", default: "http://localhost:3000" },
      profile: { type: "boolean", default: false },
      headed: { type: "boolean", default: false },
      screenshot: { type: "boolean", default: false },
      channel: { type: "string" },
      "storage-state": { type: "string" },
      list: { type: "boolean", default: false },
    },
  });
  if (values.list) {
    SCENARIOS.forEach((scenario) =>
      console.log(`${scenario.name.padEnd(28)} ${scenario.description}`),
    );
    process.exit(0);
  }
  const runs = Number(values.runs);
  const warmup = Number(values.warmup);
  if (!values.scenario) throw new Error("Pass --scenario <name> (or --list).");
  if (!Number.isInteger(runs) || runs < 1) throw new Error("Bad --runs.");
  if (!Number.isInteger(warmup) || warmup < 0) throw new Error("Bad --warmup.");
  return {
    scenario: findScenario(values.scenario),
    label: safeLabel(values.label ?? "any"),
    runs,
    warmup,
    baseURL: (values["base-url"] ?? "").replace(/\/$/, ""),
    profile: values.profile ?? false,
    headed: values.headed ?? false,
    screenshot: values.screenshot ?? false,
    channel: values.channel ?? process.env.PERF_BROWSER_CHANNEL,
    storageState: values["storage-state"],
  };
}

type Options = ReturnType<typeof parseOptions>;

function gitSha() {
  try {
    return execSync("git rev-parse HEAD", { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

interface RunArgs {
  browser: Browser;
  stream: StreamServer;
  scenario: Scenario;
  options: Options;
  run: number;
  warmup: boolean;
}

async function measureOnce({
  browser,
  stream,
  scenario,
  options,
  run,
  warmup,
}: RunArgs) {
  const context = await browser.newContext({
    baseURL: options.baseURL,
    viewport: { width: 1440, height: 900 },
    locale: "en-US",
    timezoneId: "UTC",
    storageState: options.storageState,
  });
  try {
    const userID = await signIn(
      context.request,
      options.baseURL,
      Boolean(options.storageState),
    );
    const page = await context.newPage();
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    await installRoutes(page, commonHandlers());
    const cdp = await openCDP(context, page);
    const ctx = { page, context, baseURL: options.baseURL, userID, stream };
    const interaction = await scenario.setup(ctx);
    await assertCounterLoaded(page);

    await counter(page, "reset");
    const traceWindow = await startWindow(cdp, options.profile && !warmup);
    await counter(page, "start");
    await interaction.run();
    await page.waitForTimeout(scenario.windowMs);
    await counter(page, "stop");
    const timings = await stopWindow(cdp, traceWindow);
    const heapUsedMB = await heapAfterGC(cdp, page);
    const report = await counterReport(page);
    if (warmup) return null;

    const result: RunResult = {
      scenario: scenario.name,
      label: options.label,
      run,
      startedAt: new Date().toISOString(),
      gitSha: gitSha(),
      baseURL: options.baseURL,
      windowMs: scenario.windowMs,
      metrics: {
        commits: report.commits,
        totalRenders: report.totalRenders,
        rendersPerCommit: report.rendersPerCommit,
        maxRendersPerCommit: report.maxRendersPerCommit,
        scriptMs: timings.scriptMs,
        taskMs: timings.taskMs,
        layoutMs: timings.layoutMs,
        recalcStyleMs: timings.recalcStyleMs,
        longTaskCount: timings.longTaskCount,
        longTaskMs: timings.longTaskMs,
        heapUsedMB,
      },
      components: report.components.slice(0, RANKING_LIMIT),
      causes: report.causes.slice(0, RANKING_LIMIT),
      pageErrors: pageErrors.length,
    };
    const stem = path.join(
      scenarioOutputDir(scenario.name),
      runFileName(result).replace(/\.json$/, ""),
    );
    fs.mkdirSync(path.dirname(stem), { recursive: true });
    if (timings.cpuProfile) {
      fs.writeFileSync(
        `${stem}.cpuprofile`,
        JSON.stringify(timings.cpuProfile),
      );
      result.cpuProfile = path.basename(`${stem}.cpuprofile`);
    }
    if (options.screenshot) await page.screenshot({ path: `${stem}.png` });
    const file = writeRun(scenarioOutputDir(scenario.name), result);
    console.log(`\n${formatRun(result)}\n\nsaved ${file}`);
    if (pageErrors.length) {
      console.warn(
        `⚠ ${pageErrors.length} uncaught page error(s), first: ${pageErrors[0]}`,
      );
    }
    return result;
  } finally {
    await context.close();
  }
}

async function main() {
  const options = parseOptions();
  const { scenario } = options;
  const stream = await startStreamServer();
  const browser = await chromium.launch({
    headless: !options.headed,
    channel: options.channel,
  });
  const results: RunResult[] = [];
  try {
    const total = options.warmup + options.runs;
    for (let index = 1; index <= total; index++) {
      const warmup = index <= options.warmup;
      const run = index - options.warmup;
      console.log(
        warmup
          ? `\n▶ ${scenario.name} · warm-up ${index}/${options.warmup} (not saved)`
          : `\n▶ ${scenario.name} · ${options.label} · run ${run}/${options.runs}`,
      );
      const args = { browser, stream, scenario, options, run, warmup };
      const result = await measureOnce(args);
      if (result) results.push(result);
    }
  } finally {
    await browser.close();
    await stream.close();
  }
  if (results.length > 1) {
    console.log(`\n${formatSummary({ label: options.label, runs: results })}`);
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});

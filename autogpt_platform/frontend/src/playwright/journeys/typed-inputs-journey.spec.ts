import type { Locator, Page } from "@playwright/test";
import { expect, test } from "./fixtures";
import { pinKnownFailure } from "./known-failure";
import {
  type Agent,
  createTypedInputsAgent,
  download,
  escapeRegExp,
  fileOutputContent,
  fillTable,
  inputField,
  isRunStart,
  longTextField,
  observe,
  openRunDialog,
  pickDropdownOption,
  pickTime,
  readShownOutputs,
  rerunTask,
  setToggle,
  showsText,
  shownOutput,
  startTask,
  type TypedInput,
  uploadFile,
  waitForRunToEnd,
} from "./typed-inputs";

// What people type, picked for what tends to break: spaces at either end,
// line breaks, scripts other than Latin, emoji, and text that looks like a
// number or JSON.
const SHORT_TEXT = "  Zoë’s café — 東京 🍣  ";
const LONG_TEXT =
  "  Hi team,\n\nПривет! שלום — 你好 👋🏽\n    an indented line\nthe last line, with spaces after   ";
const SHIPPING = "Express — next day";
const TABLE = [
  { Item: "Notebook", Price: "4.50" },
  { Item: "Pen", Price: "0" },
];
const FILE = {
  name: "notes.txt",
  mimeType: "text/plain",
  buffer: Buffer.from("Grüße aus Zürich 👋\nsecond line\n"),
};
const DATE = "2026-03-15";
const TIME = { hour: "02", minute: "30", meridiem: "PM" } as const;
// What the workflow gets for 2:30 PM.
const TIME_GOT = "14:30:00";
// A known failure must end as a failed assertion, not a timeout, so its
// check waits well inside the test's budget.
const PINNED_CHECK_TIMEOUT = 5_000;

// Optional: a picked date never counts as filled in the run form, so a
// required one keeps Start Task disabled (#17304). A test of its own pins
// that; the others still get to run a date through.
const OPTIONAL_DATE: TypedInput = {
  name: "date",
  kind: "date",
  settings: { advanced: true },
};

const EVERY_INPUT: TypedInput[] = [
  { name: "short_text", kind: "short text" },
  { name: "long_text", kind: "long text" },
  { name: "digits_text", kind: "short text" },
  { name: "boolean_text", kind: "short text" },
  { name: "exponent_text", kind: "short text" },
  { name: "zero", kind: "number" },
  { name: "negative", kind: "number" },
  OPTIONAL_DATE,
  { name: "time", kind: "time" },
  {
    name: "shipping",
    kind: "dropdown",
    settings: { options: ["Standard", SHIPPING, "Pick up in store"] },
  },
  { name: "toggle_on", kind: "toggle" },
  { name: "toggle_off", kind: "toggle" },
  {
    name: "table",
    kind: "table",
    settings: { column_headers: ["Item", "Price"] },
  },
  { name: "file", kind: "file" },
];

// What the workflow should get from each input but the file, with its type.
const GOT: Record<string, unknown> = {
  short_text: SHORT_TEXT,
  long_text: LONG_TEXT,
  digits_text: "00123",
  boolean_text: "true",
  exponent_text: "1e5",
  zero: 0,
  negative: -42,
  date: DATE,
  time: TIME_GOT,
  shipping: SHIPPING,
  toggle_on: true,
  toggle_off: false,
  table: TABLE,
};

// What the run view should show for each, the table, file and 1e5 aside.
const SHOWN: Record<string, string> = {
  short_text: SHORT_TEXT,
  long_text: LONG_TEXT,
  digits_text: "00123",
  boolean_text: "true",
  zero: "0",
  negative: "-42",
  date: DATE,
  time: TIME_GOT,
  shipping: SHIPPING,
  toggle_on: "true",
  toggle_off: "false",
};

test.describe("typed inputs journey", () => {
  test.describe("in UTC", () => {
    // A date is checked outside UTC on its own, below.
    test.use({ timezoneId: "UTC" });

    test("every kind of input reaches the workflow as it was entered, and the run view and a rerun show each value unchanged", async ({
      page,
      journeyUser: _user,
    }, testInfo) => {
      const agent = await createTypedInputsAgent(page, EVERY_INPUT);
      const dialog = await openRunDialog(page, agent);
      await fillEveryInput(page, dialog);
      const { runID, sent } = await startTask(page, agent, dialog);
      observe(testInfo, `the run form sent ${JSON.stringify(sent)}`);

      const run = await waitForRunToEnd(page, agent, runID);
      observe(
        testInfo,
        `the run ended ${run.status} with outputs ${JSON.stringify(run.outputs)}`,
      );
      expect(run.status, "the run should complete").toBe("COMPLETED");
      const { file, ...got } = run.outputs;
      expect(
        got,
        "the workflow should get every value as it was entered, with its type",
      ).toEqual(asOutputs(GOT));
      await expectFileOutput(page, file);
      await expectEveryOutputShown(page);
      observe(
        testInfo,
        `the run view shows ${JSON.stringify(await readShownOutputs(page))}`,
      );
      await expectRunViewShowsEveryValue(page);

      // A rerun sends the run's stored inputs again.
      const rerun = await rerunTask(page, agent);
      const again = await waitForRunToEnd(page, agent, rerun.runID);
      expect(again.status, "the rerun should complete").toBe("COMPLETED");
      const { file: fileAgain, ...gotAgain } = again.outputs;
      expect(
        gotAgain,
        "a rerun should give the workflow the same values",
      ).toEqual(got);
      await expectFileOutput(page, fileAgain);
      await expectEveryOutputShown(page);
      await expectRunViewShowsEveryValue(page);

      // Text that JSON.parse can read (here, as a number) is shown as typed.
      expect(
        (await readShownOutputs(page)).exponent_text,
        "the run view should show the text 1e5 as typed",
      ).toBe("1e5");
    });

    test(
      "a task whose workflow needs a date can start once the date is picked",
      { annotation: issue(17304) },
      async ({ page, journeyUser: _user }) => {
        const agent = await createTypedInputsAgent(page, [
          { name: "date", kind: "date" },
        ]);
        const dialog = await openRunDialog(page, agent);
        await inputField(dialog, "date").fill(DATE);
        await expect(
          inputField(dialog, "date"),
          "the form should show the date picked",
        ).toHaveValue(DATE);

        const pinned = pinKnownFailure(
          "#17304: a picked date doesn't count as filled, so Start Task stays disabled",
          "Start Task should be enabled once the date is picked",
        );
        await expect(
          dialog.getByRole("button", { name: "Start Task", exact: true }),
          pinned,
        ).toBeEnabled({ timeout: PINNED_CHECK_TIMEOUT });
      },
    );
  });

  test.describe("a decimal number", () => {
    // A fix may refuse 1180.5 when the task starts: the refusal is then what
    // the journey expects, not an error.
    test.use({
      expectedResponses: [400, 422].map((status) => ({
        status,
        url: "/execute/",
        method: "POST",
      })),
    });

    test(
      "a decimal in a Number input reaches the workflow as typed, or is refused with an error the user sees, and is never cut to a whole number",
      { annotation: issue(17259) },
      async ({ page, journeyUser: _user }, testInfo) => {
        const agent = await createTypedInputsAgent(page, [
          { name: "amount", kind: "number" },
        ]);
        const dialog = await openRunDialog(page, agent);
        await inputField(dialog, "amount").fill("1180.5");

        const outcome = await startWithDecimal(page, agent, dialog);
        observe(testInfo, `1180.5 in a Number input: ${outcome}`);
        const pinned = pinKnownFailure(
          "#17259: a Number input cuts 1180.5 to 1180 and says nothing",
          "1180.5 should reach the workflow as typed, or be refused with an error the user sees",
        );
        expect(outcome, pinned).toMatch(/^(kept|refused)\b/);
      },
    );
  });

  for (const timezoneId of ["America/Los_Angeles", "Pacific/Auckland"]) {
    test.describe(`in ${timezoneId}`, () => {
      test.use({ timezoneId });

      test(`a date and a time picked in ${timezoneId} reach the workflow as picked`, async ({
        page,
        journeyUser: _user,
      }, testInfo) => {
        const agent = await createTypedInputsAgent(page, [
          OPTIONAL_DATE,
          { name: "time", kind: "time" },
        ]);
        const dialog = await openRunDialog(page, agent);
        await inputField(dialog, "date").fill(DATE);
        await expect(
          inputField(dialog, "date"),
          "the form should show the date picked",
        ).toHaveValue(DATE);
        await pickTime(page, dialog, TIME);
        const { runID, sent } = await startTask(page, agent, dialog);
        observe(testInfo, `the run form sent ${JSON.stringify(sent)}`);

        const run = await waitForRunToEnd(page, agent, runID);
        observe(
          testInfo,
          `the run ended ${run.status} with outputs ${JSON.stringify(run.outputs)}`,
        );
        expect(
          run.outputs.time,
          "the workflow should get the time picked",
        ).toEqual([TIME_GOT]);
        await expect
          .poll(async () => (await readShownOutputs(page)).time, {
            message: "the run view should show the time picked",
          })
          .toBe(TIME_GOT);
        const failures = page.getByTestId("run-failures");
        observe(
          testInfo,
          `the run view's failed steps: ${(await failures.count()) ? await failures.innerText() : "none"}`,
        );

        const shown = await readShownOutputs(page);
        expect(
          { status: run.status, got: run.outputs.date, shown: shown.date },
          "the workflow should get the date picked, and the run view show it",
        ).toEqual({ status: "COMPLETED", got: [DATE], shown: DATE });
      });
    });
  }
});

async function fillEveryInput(page: Page, dialog: Locator) {
  await inputField(dialog, "short_text").fill(SHORT_TEXT);
  await longTextField(dialog, "long_text").fill(LONG_TEXT);
  await inputField(dialog, "digits_text").fill("00123");
  await inputField(dialog, "boolean_text").fill("true");
  await inputField(dialog, "exponent_text").fill("1e5");
  await inputField(dialog, "zero").fill("0");
  await inputField(dialog, "negative").fill("-42");
  await inputField(dialog, "date").fill(DATE);
  await pickTime(page, dialog, TIME);
  await pickDropdownOption(page, dialog, "shipping", SHIPPING);
  await setToggle(dialog, "toggle_on", true);
  await setToggle(dialog, "toggle_off", false);
  await fillTable(dialog, TABLE);
  await uploadFile(page, dialog, FILE);
}

// The file output points to the uploaded file's own content.
async function expectFileOutput(page: Page, file: unknown[] | undefined) {
  expect(file, "the file input should hand on one file").toHaveLength(1);
  expect(
    (await fileOutputContent(page, file?.[0])).toString("utf8"),
    "the file output should hold the uploaded content",
  ).toBe(FILE.buffer.toString("utf8"));
}

// Every output has its place in the run view, falsy ones included.
async function expectEveryOutputShown(page: Page) {
  const names = EVERY_INPUT.map((input) => input.name).sort();
  await expect
    .poll(async () => Object.keys(await readShownOutputs(page)).sort(), {
      message: "the run view should show every output",
    })
    .toEqual(names);
}

async function expectRunViewShowsEveryValue(page: Page) {
  await expect
    .poll(async () => pick(await readShownOutputs(page), Object.keys(SHOWN)), {
      message: "the run view should show every value as it was entered",
    })
    .toEqual(SHOWN);
  await expect(
    shownOutput(page, "table"),
    "the run view should show each row of the table, cell by cell",
  ).toHaveText(tableCells(TABLE));
  const link = shownOutput(page, "file").locator("a[download]");
  await expect(
    link,
    "the run view should link to the file to download",
  ).toHaveAttribute("href", /\/download$/);
  expect(
    (await download(page, String(await link.getAttribute("href")))).toString(
      "utf8",
    ),
    "the run view's file link should download the uploaded content",
  ).toBe(FILE.buffer.toString("utf8"));
}

// Each row's cells in order, as "column": "value".
function tableCells(rows: Record<string, string>[]) {
  return new RegExp(
    rows
      .flatMap((row) =>
        Object.entries(row).map(
          ([column, value]) =>
            `"${escapeRegExp(column)}":\\s*"${escapeRegExp(value)}"`,
        ),
      )
      .join("[\\s\\S]*"),
  );
}

// An error a refusal of 1180.5 would name: the value, or why it was refused.
const DECIMAL_REFUSED = /1180\.5|\bint(eger)?\b|whole number|decimal|fraction/i;

/**
 * Starts a task with 1180.5 in the Number input, and says what became of it:
 * "kept", "refused ..." with an error the user sees, or what happened instead.
 */
async function startWithDecimal(page: Page, agent: Agent, dialog: Locator) {
  const everywhere = page.locator("body");
  const start = dialog.getByRole("button", { name: "Start Task", exact: true });
  if (await start.isDisabled()) {
    return (await showsText(dialog, DECIMAL_REFUSED))
      ? "refused in the form"
      : "Start Task is disabled, and nothing says why";
  }
  const response = page
    .waitForResponse(isRunStart(agent.graphID), { timeout: 10_000 })
    .catch(() => null);
  await start.click();
  const started = await response;
  if (!started) {
    return (await showsText(everywhere, DECIMAL_REFUSED))
      ? "refused in the form"
      : "Start Task started nothing, and nothing says why";
  }
  if (!started.ok()) {
    return (await showsText(everywhere, DECIMAL_REFUSED))
      ? "refused as the task started"
      : `refused with a ${started.status()}, and nothing says why`;
  }
  const { id: runID } = (await started.json()) as { id: string };
  const run = await waitForRunToEnd(page, agent, runID);
  const got = run.outputs.amount;
  if (got === undefined) {
    return (await showsText(page.getByTestId("run-failures"), DECIMAL_REFUSED))
      ? "refused as the run reached it"
      : `the run ended ${run.status} without the number, and nothing says why`;
  }
  await expect
    .poll(async () => (await readShownOutputs(page)).amount, {
      message: "the run view should show the number the workflow got",
    })
    .toBeDefined();
  const shown = (await readShownOutputs(page)).amount;
  if (JSON.stringify(got) === "[1180.5]" && shown === "1180.5") return "kept";
  return `the workflow got ${JSON.stringify(got)} and the run view shows ${JSON.stringify(shown)}`;
}

function issue(number: number) {
  return {
    type: "issue",
    description: `https://github.com/Significant-Gravitas/AutoGPT/issues/${number}`,
  };
}

function asOutputs(values: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(values).map(([name, value]) => [name, [value]]),
  );
}

function pick(values: Record<string, string>, names: string[]) {
  return Object.fromEntries(names.map((name) => [name, values[name]]));
}

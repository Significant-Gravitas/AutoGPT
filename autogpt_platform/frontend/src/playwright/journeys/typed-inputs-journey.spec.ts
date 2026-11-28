import type { Locator, Page } from "@playwright/test";
import { expect, test } from "./fixtures";
import { pinKnownFailure } from "./known-failure";
import {
  type Agent,
  createTypedInputsAgent,
  escapeRegExp,
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
  shownOutput,
  showsWithin,
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
const FILE_TEXT = "Grüße aus Zürich 👋\nsecond line\n";
const FILE = {
  name: "notes.txt",
  mimeType: "text/plain",
  buffer: Buffer.from(FILE_TEXT),
};
const DATE = "2026-03-15";
const TIME = { hour: "02", minute: "30", meridiem: "PM" } as const;
// What the workflow gets for 2:30 PM.
const TIME_GOT = "14:30:00";
// Short text that reads as something else: digits with leading zeros, a
// JSON boolean and two JSON numbers.
const LOOKS_LIKE_DATA = {
  digits_text: "00123",
  boolean_text: "true",
  exponent_text: "1e5",
  price_text: "1.50",
};
const DATE_INPUT: TypedInput = { name: "date", kind: "date" };

const EVERY_INPUT: TypedInput[] = [
  { name: "short_text", kind: "short text" },
  { name: "long_text", kind: "long text" },
  { name: "zero", kind: "number" },
  { name: "negative", kind: "number" },
  DATE_INPUT,
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
  zero: 0,
  negative: -42,
  date: DATE,
  time: TIME_GOT,
  shipping: SHIPPING,
  toggle_on: true,
  toggle_off: false,
  table: TABLE,
};

// What the run view should show for each, the table and the file aside.
const SHOWN: Record<string, string> = {
  short_text: SHORT_TEXT,
  long_text: LONG_TEXT,
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
      expectOneFile(file);
      await expectEveryOutputShown(page, EVERY_INPUT);
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
      expectOneFile(fileAgain);
      await expectEveryOutputShown(page, EVERY_INPUT);
      await expectRunViewShowsEveryValue(page);
      const link = await fileLink(page);

      const response = await page.request.get(link);
      expect(
        {
          status: response.status(),
          content: (await response.body()).toString("utf8"),
        },
        "the run view's file link should download the uploaded content",
      ).toEqual({ status: 200, content: FILE_TEXT });
    });

    test(
      "text that looks like a number or JSON reaches the workflow, and shows in the run view, exactly as typed",
      { annotation: issue(17307) },
      async ({ page, journeyUser: _user }, testInfo) => {
        const inputs = Object.keys(LOOKS_LIKE_DATA).map(
          (name): TypedInput => ({ name, kind: "short text" }),
        );
        const agent = await createTypedInputsAgent(page, inputs);
        const dialog = await openRunDialog(page, agent);
        for (const [name, text] of Object.entries(LOOKS_LIKE_DATA)) {
          await inputField(dialog, name).fill(text);
        }
        const { runID } = await startTask(page, agent, dialog);

        const run = await waitForRunToEnd(page, agent, runID);
        expect(run.status, "the run should complete").toBe("COMPLETED");
        expect(
          run.outputs,
          "the workflow should get each text exactly as typed",
        ).toEqual(asOutputs(LOOKS_LIKE_DATA));
        await expectEveryOutputShown(page, inputs);
        const shown = await readShownOutputs(page);
        observe(testInfo, `the run view shows ${JSON.stringify(shown)}`);
        expect(
          pick(shown, Object.keys(LOOKS_LIKE_DATA)),
          "the run view should show each text exactly as typed",
        ).toEqual(LOOKS_LIKE_DATA);
      },
    );

    test(
      "a task whose workflow needs a date can start once the date is picked",
      { annotation: issue(17304) },
      async ({ page, journeyUser: _user }) => {
        const agent = await createTypedInputsAgent(page, [DATE_INPUT]);
        const dialog = await openRunDialog(page, agent);
        await inputField(dialog, "date").fill(DATE);
        await expect(
          inputField(dialog, "date"),
          "the form should show the date picked",
        ).toHaveValue(DATE);

        await expect(
          dialog.getByRole("button", { name: "Start Task", exact: true }),
          "Start Task should be enabled once the date is picked",
        ).toBeEnabled();
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
        // Only the cut is the known bug: anything else, such as a refusal
        // that says nothing, fails here rather than under the pin.
        expect(
          outcome,
          "1180.5 should be kept, refused with an error the user sees, or, as #17259 does, cut to a whole number",
        ).toMatch(/^(kept|refused|cut to a whole number)\b/);
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

      test(
        `a date and a time picked in ${timezoneId} reach the workflow as picked`,
        { annotation: issue(17305) },
        async ({ page, journeyUser: _user }, testInfo) => {
          const agent = await createTypedInputsAgent(page, [
            DATE_INPUT,
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
          expect(
            { status: run.status, got: run.outputs.date },
            "the workflow should get the date picked",
          ).toEqual({ status: "COMPLETED", got: [DATE] });
          // The run view can show one output a moment before the other, so
          // wait for both rather than read the date once the time shows.
          await expect
            .poll(
              async () => {
                const shown = await readShownOutputs(page);
                return { date: shown.date, time: shown.time };
              },
              {
                message:
                  "the run view should show the date and the time picked",
              },
            )
            .toEqual({ date: DATE, time: TIME_GOT });
        },
      );
    });
  }
});

async function fillEveryInput(page: Page, dialog: Locator) {
  await inputField(dialog, "short_text").fill(SHORT_TEXT);
  await longTextField(dialog, "long_text").fill(LONG_TEXT);
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

// The file input hands on one reference to the uploaded file.
function expectOneFile(file: unknown[] | undefined) {
  expect(file, "the file input should hand on one file").toHaveLength(1);
  expect(
    String(file?.[0]),
    "the file output should be a workspace file or a data URI",
  ).toMatch(/^(workspace:\/\/|data:)/);
}

// Every output has its place in the run view, falsy ones included.
async function expectEveryOutputShown(page: Page, inputs: TypedInput[]) {
  const names = inputs.map((input) => input.name).sort();
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
}

// The run view's link to download the file output.
async function fileLink(page: Page) {
  const link = shownOutput(page, "file").locator("a[download]");
  await expect(
    link,
    "the run view should link to the file to download",
  ).toHaveAttribute("href", /\/download$/);
  return String(await link.getAttribute("href"));
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
// How long the app may take to answer Start Task, by starting the run or
// saying why it won't, on a loaded runner.
const ANSWER_TIMEOUT = 30_000;
// How long a step of the form, or a message once the app has answered, may
// take to show: the journeys' expect timeout.
const SHOW_TIMEOUT = 15_000;

/**
 * Starts a task with 1180.5 in the Number input, and says what became of it:
 * "kept", "refused ..." with an error the user sees, "cut to a whole number"
 * (#17259), or what happened instead.
 */
async function startWithDecimal(page: Page, agent: Agent, dialog: Locator) {
  const refusal = decimalRefusal(page, dialog);
  const start = dialog.getByRole("button", { name: "Start Task", exact: true });
  // A check of the number may disable Start Task a render after the typing:
  // wait for the form to settle on Start Task enabled, or a refusal shown.
  const enabled = dialog.getByRole("button", {
    name: "Start Task",
    exact: true,
    disabled: false,
  });
  if (!(await showsWithin(enabled.or(refusal).first(), SHOW_TIMEOUT))) {
    return "Start Task is disabled, and nothing says why";
  }
  if (await refusal.isVisible()) return "refused in the form";

  const response = page
    .waitForResponse(isRunStart(agent.graphID), { timeout: ANSWER_TIMEOUT })
    .catch(() => null);
  if (!(await clicks(start))) {
    return (await showsWithin(refusal, SHOW_TIMEOUT))
      ? "refused in the form"
      : "Start Task can't be pressed, and nothing says why";
  }
  // The form may refuse the number without starting anything.
  const started = await Promise.race([
    response,
    showsWithin(refusal, ANSWER_TIMEOUT).then((shown) =>
      shown ? ("refused in the form" as const) : null,
    ),
  ]);
  if (!started) return "Start Task started nothing, and nothing says why";
  if (typeof started === "string") return started;
  if (!started.ok()) {
    return (await showsWithin(refusal, SHOW_TIMEOUT))
      ? "refused as the task started"
      : `the task returned ${started.status()}, and nothing says why`;
  }
  const { id: runID } = (await started.json()) as { id: string };
  const run = await waitForRunToEnd(page, agent, runID);
  const got = run.outputs.amount;
  if (got === undefined) {
    return (await showsWithin(refusal, SHOW_TIMEOUT))
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
  const [number] = got;
  if (
    got.length === 1 &&
    isWholeNumberCut(number) &&
    shown === String(number)
  ) {
    return `cut to a whole number: the workflow got ${number}, and the run view shows it`;
  }
  return `the workflow got ${JSON.stringify(got)} and the run view shows ${JSON.stringify(shown)}`;
}

// A refusal the user sees: in the run form, an alert or a toast, or the run's
// failed steps, never text elsewhere on the page.
function decimalRefusal(page: Page, dialog: Locator) {
  return dialog
    .or(page.getByRole("alert"))
    .or(page.getByRole("status"))
    .or(page.locator("[data-sonner-toast]"))
    .or(page.getByTestId("run-failures"))
    .getByText(DECIMAL_REFUSED)
    .filter({ visible: true })
    .first();
}

// Clicks `button`, or gives up once it has stayed disabled for a while.
function clicks(button: Locator) {
  return button.click({ timeout: SHOW_TIMEOUT }).then(
    () => true,
    () => false,
  );
}

// 1180.5 without its decimals: 1180, or 1181 if rounded.
function isWholeNumberCut(value: unknown) {
  return value === 1180 || value === 1181;
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

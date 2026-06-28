import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const toastSpy = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: toastSpy }),
  toast: toastSpy,
}));

// The component drives a generated react-query mutation hook. We replace it
// with a controllable stub: `mutate` invokes whichever callback the active
// test wired up, so we can exercise the success / error / non-200 branches
// without a live backend or generated client.
type MutationCallbacks = {
  onSuccess: (response: { status: number; data: unknown }) => void;
  onError: (error: unknown) => void;
};

const mutationState = vi.hoisted(() => ({
  run: (_callbacks: MutationCallbacks) => {},
  isPending: false,
}));
const mutateSpy = vi.hoisted(() => vi.fn());

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  usePostV2GenerateTestData: (options: { mutation: MutationCallbacks }) => ({
    mutate: (variables: unknown) => {
      mutateSpy(variables);
      mutationState.run(options.mutation);
    },
    isPending: mutationState.isPending,
  }),
}));

import { GenerateTestDataButton } from "../GenerateTestDataButton";

function openDialog() {
  fireEvent.click(screen.getByRole("button", { name: "Generate Test Data" }));
}

async function clickGenerate() {
  // Once the dialog is open there are two "Generate Test Data" buttons: the
  // page trigger and the dialog footer action (portaled, so last in the DOM).
  await waitFor(() =>
    expect(
      screen.getByText(/populate the database with sample test data/i),
    ).toBeDefined(),
  );
  const actions = screen.getAllByRole("button", { name: "Generate Test Data" });
  fireEvent.click(actions[actions.length - 1]);
}

beforeEach(() => {
  toastSpy.mockReset();
  mutateSpy.mockReset();
  mutationState.run = () => {};
  mutationState.isPending = false;
});

afterEach(() => {
  cleanup();
});

describe("GenerateTestDataButton", () => {
  test("renders the trigger button", () => {
    render(<GenerateTestDataButton />);
    expect(
      screen.getByRole("button", { name: "Generate Test Data" }),
    ).toBeDefined();
  });

  test("opens a dialog with the script type selector and warning", async () => {
    render(<GenerateTestDataButton />);
    openDialog();
    await waitFor(() => {
      expect(screen.getByText("Script Type")).toBeDefined();
      expect(
        screen.getByText(/disabled in production environments/i),
      ).toBeDefined();
    });
  });

  test("submits with the default script type and shows a success result", async () => {
    mutationState.run = (callbacks) =>
      callbacks.onSuccess({
        status: 200,
        data: {
          success: true,
          message: "E2E test data generated successfully",
          details: { users_created: 15 },
        },
      });

    render(<GenerateTestDataButton />);
    openDialog();
    await clickGenerate();

    expect(mutateSpy).toHaveBeenCalledWith({ data: { script_type: "e2e" } });
    await waitFor(() =>
      expect(
        screen.getByText("E2E test data generated successfully"),
      ).toBeDefined(),
    );
    // Details map is rendered with underscores replaced by spaces.
    expect(screen.getByText(/users created/i)).toBeDefined();
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Success" }),
    );
  });

  test("shows a destructive toast when the API reports failure", async () => {
    mutationState.run = (callbacks) =>
      callbacks.onSuccess({
        status: 200,
        data: {
          success: false,
          message: "Test data generation is disabled in production environments.",
        },
      });

    render(<GenerateTestDataButton />);
    openDialog();
    await clickGenerate();

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Error", variant: "destructive" }),
      ),
    );
    expect(
      screen.getByText(/disabled in production environments/i),
    ).toBeDefined();
  });

  test("surfaces a generic error when the mutation throws", async () => {
    mutationState.run = (callbacks) =>
      callbacks.onError(new Error("network blip"));

    render(<GenerateTestDataButton />);
    openDialog();
    await clickGenerate();

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Error", variant: "destructive" }),
      ),
    );
    expect(
      screen.getByText(/Failed to generate test data: network blip/i),
    ).toBeDefined();
  });

  test("ignores non-200 success responses", async () => {
    mutationState.run = (callbacks) =>
      callbacks.onSuccess({ status: 401, data: { detail: "unauthorized" } });

    render(<GenerateTestDataButton />);
    openDialog();
    await clickGenerate();

    expect(mutateSpy).toHaveBeenCalledTimes(1);
    expect(toastSpy).not.toHaveBeenCalled();
  });
});

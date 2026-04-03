import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { UseFormReturn, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { renderHook } from "@testing-library/react";
import { useControlPanelStore } from "../stores/controlPanelStore";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { NewSaveControl } from "../components/NewControlPanel/NewSaveControl/NewSaveControl";
import { useNewSaveControl } from "../components/NewControlPanel/NewSaveControl/useNewSaveControl";

const formSchema = z.object({
  name: z.string().min(1, "Name is required").max(100),
  description: z.string().max(500),
});

type SaveableGraphFormValues = z.infer<typeof formSchema>;

const mockHandleSave = vi.fn();

vi.mock(
  "../components/NewControlPanel/NewSaveControl/useNewSaveControl",
  () => ({
    useNewSaveControl: vi.fn(),
  }),
);

const mockUseNewSaveControl = vi.mocked(useNewSaveControl);

function createMockForm(
  defaults: SaveableGraphFormValues = { name: "", description: "" },
): UseFormReturn<SaveableGraphFormValues> {
  const { result } = renderHook(() =>
    useForm<SaveableGraphFormValues>({
      resolver: zodResolver(formSchema),
      defaultValues: defaults,
    }),
  );
  return result.current;
}

function setupMock(overrides: {
  isSaving?: boolean;
  graphVersion?: number;
  name?: string;
  description?: string;
}) {
  const form = createMockForm({
    name: overrides.name ?? "",
    description: overrides.description ?? "",
  });

  mockUseNewSaveControl.mockReturnValue({
    form,
    isSaving: overrides.isSaving ?? false,
    graphVersion: overrides.graphVersion,
    handleSave: mockHandleSave,
  });

  return form;
}

function resetStore() {
  useControlPanelStore.setState({
    blockMenuOpen: false,
    saveControlOpen: false,
    forceOpenBlockMenu: false,
    forceOpenSave: false,
  });
}

beforeEach(() => {
  cleanup();
  resetStore();
  mockHandleSave.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("NewSaveControl", () => {
  it("renders save button trigger", () => {
    setupMock({});
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    expect(screen.getByTestId("save-control-save-button")).toBeDefined();
  });

  it("renders name and description inputs when popover is open", () => {
    useControlPanelStore.setState({ saveControlOpen: true });
    setupMock({});
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    expect(screen.getByTestId("save-control-name-input")).toBeDefined();
    expect(screen.getByTestId("save-control-description-input")).toBeDefined();
  });

  it("does not render popover content when closed", () => {
    useControlPanelStore.setState({ saveControlOpen: false });
    setupMock({});
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    expect(screen.queryByTestId("save-control-name-input")).toBeNull();
    expect(screen.queryByTestId("save-control-description-input")).toBeNull();
  });

  it("shows version output when graphVersion is set", () => {
    useControlPanelStore.setState({ saveControlOpen: true });
    setupMock({ graphVersion: 3 });
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    const versionInput = screen.getByTestId("save-control-version-output");
    expect(versionInput).toBeDefined();
    expect((versionInput as HTMLInputElement).disabled).toBe(true);
  });

  it("hides version output when graphVersion is undefined", () => {
    useControlPanelStore.setState({ saveControlOpen: true });
    setupMock({ graphVersion: undefined });
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    expect(screen.queryByTestId("save-control-version-output")).toBeNull();
  });

  it("enables save button when isSaving is false", () => {
    useControlPanelStore.setState({ saveControlOpen: true });
    setupMock({ isSaving: false });
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    const saveButton = screen.getByTestId("save-control-save-agent-button");
    expect((saveButton as HTMLButtonElement).disabled).toBe(false);
  });

  it("disables save button when isSaving is true", () => {
    useControlPanelStore.setState({ saveControlOpen: true });
    setupMock({ isSaving: true });
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    const saveButton = screen.getByRole("button", { name: /save agent/i });
    expect((saveButton as HTMLButtonElement).disabled).toBe(true);
  });

  it("calls handleSave on form submission with valid data", async () => {
    useControlPanelStore.setState({ saveControlOpen: true });
    const form = setupMock({ name: "My Agent", description: "A description" });

    form.setValue("name", "My Agent");
    form.setValue("description", "A description");

    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    const saveButton = screen.getByTestId("save-control-save-agent-button");
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockHandleSave).toHaveBeenCalledWith(
        { name: "My Agent", description: "A description" },
        expect.anything(),
      );
    });
  });

  it("does not call handleSave when name is empty (validation fails)", async () => {
    useControlPanelStore.setState({ saveControlOpen: true });
    setupMock({ name: "", description: "" });
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    const saveButton = screen.getByTestId("save-control-save-agent-button");
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockHandleSave).not.toHaveBeenCalled();
    });
  });

  it("popover stays open when forceOpenSave is true", () => {
    useControlPanelStore.setState({
      saveControlOpen: false,
      forceOpenSave: true,
    });
    setupMock({});
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    expect(screen.getByTestId("save-control-name-input")).toBeDefined();
  });

  it("allows typing in name and description inputs", () => {
    useControlPanelStore.setState({ saveControlOpen: true });
    setupMock({});
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    const nameInput = screen.getByTestId(
      "save-control-name-input",
    ) as HTMLInputElement;
    const descriptionInput = screen.getByTestId(
      "save-control-description-input",
    ) as HTMLInputElement;

    fireEvent.change(nameInput, { target: { value: "Test Agent" } });
    fireEvent.change(descriptionInput, {
      target: { value: "Test Description" },
    });

    expect(nameInput.value).toBe("Test Agent");
    expect(descriptionInput.value).toBe("Test Description");
  });

  it("displays save button text", () => {
    useControlPanelStore.setState({ saveControlOpen: true });
    setupMock({});
    render(
      <TooltipProvider>
        <NewSaveControl />
      </TooltipProvider>,
    );

    expect(screen.getByText("Save Agent")).toBeDefined();
  });
});

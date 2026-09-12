import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { LlmModelPicker } from "../components/LlmModelPicker";
import { LlmModelMetadata } from "../types";

const sonnet: LlmModelMetadata = {
  name: "claude-sonnet-4-6",
  title: "Claude Sonnet 4.6",
  creator: "anthropic",
  creator_name: "Anthropic",
  provider: "anthropic",
  provider_name: "Anthropic",
  price_tier: 3,
};

describe("LlmModelPicker trigger", () => {
  test("shows the selected model's display name", () => {
    render(
      <LlmModelPicker
        models={[sonnet]}
        selectedName={sonnet.name}
        selectedModel={sonnet}
        recommendedModel={sonnet}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText("Claude Sonnet 4.6")).toBeDefined();
  });

  test("shows the raw stored slug when the model is no longer available", () => {
    // A hidden/kill-switched model drops out of llm_model_metadata while the
    // node still stores (and executes) its slug — the trigger must show the
    // truth, never silently substitute the recommended model.
    render(
      <LlmModelPicker
        models={[sonnet]}
        selectedName="retired/old-model"
        selectedModel={undefined}
        recommendedModel={sonnet}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText("retired/old-model (unavailable)")).toBeDefined();
    expect(screen.queryByText("Claude Sonnet 4.6")).toBeNull();
  });

  test("falls back to the recommended model only when nothing is stored", () => {
    render(
      <LlmModelPicker
        models={[sonnet]}
        selectedName=""
        selectedModel={undefined}
        recommendedModel={sonnet}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText("Claude Sonnet 4.6")).toBeDefined();
  });
});

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  MASKED_IN_REPLAY,
  UNMASKED_IN_REPLAY,
  fixedCopy,
} from "@/lib/replay-privacy";
import {
  UnmaskedContainer,
  recordReplayText,
} from "@/tests/integrations/sentry-replay";
import { render } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

afterEach(cleanup);

// Recorded with Sentry's own replay code (rrweb), so these pin down how it
// resolves nested markers, not how we assume it does.
describe("Sentry replay masking", () => {
  it("masks text that no element opts in", async () => {
    render(<p>Alice Example</p>);

    expect(await recordReplayText()).not.toContain("Alice");
  });

  it("shows fixed copy only where the atom opts in", async () => {
    render(
      <>
        <Text variant="body" unmask>
          Workflows
        </Text>
        <Text variant="body">Quarterly report</Text>
        <Button unmask aria-label="Save workflow">
          Save
        </Button>
        <Button aria-label="Delete Quarterly report">Delete</Button>
      </>,
    );

    const recorded = await recordReplayText();
    expect(recorded).toContain("Workflows");
    expect(recorded).toContain("Save");
    expect(recorded).toContain("Save workflow");
    expect(recorded).not.toContain("Quarterly report");
    expect(recorded).not.toContain("Delete");
  });

  // This is why only the element that holds the fixed string opts in: an
  // unmasked wrapper would show everything inside it.
  it("shows unmarked text inside an unmasked element", async () => {
    render(
      <UnmaskedContainer>
        <span>Shown by its ancestor</span>
      </UnmaskedContainer>,
    );

    expect(await recordReplayText()).toContain("Shown by its ancestor");
  });

  it("keeps masked content masked inside an unmasked element, at any depth", async () => {
    render(
      <UnmaskedContainer>
        <span>Recent chats</span>
        <div className={MASKED_IN_REPLAY}>
          <p>
            Plan my trip to <strong>Lisbon</strong>
          </p>
          <span className={UNMASKED_IN_REPLAY}>Rename</span>
        </div>
      </UnmaskedContainer>,
    );

    const recorded = await recordReplayText();
    expect(recorded).toContain("Recent chats");
    expect(recorded).not.toContain("Plan my trip");
    expect(recorded).not.toContain("Lisbon");
    // A nearer opt-in still wins, so it stays on fixed copy too.
    expect(recorded).toContain("Rename");
  });

  it("masks what is typed into a field even inside an unmasked element", async () => {
    render(
      <UnmaskedContainer>
        <input defaultValue="alice@example.com" aria-label="Email" />
        <textarea defaultValue="My secret notes" />
        <select defaultValue="acme">
          <option value="acme">Acme Corp</option>
        </select>
        <div contentEditable suppressContentEditableWarning>
          <p>Draft to Bob</p>
        </div>
        <input
          className={UNMASKED_IN_REPLAY}
          defaultValue="hunter2"
          placeholder="Search workflows"
        />
      </UnmaskedContainer>,
    );

    const recorded = await recordReplayText();
    expect(recorded).not.toContain("alice@example.com");
    expect(recorded).not.toContain("My secret notes");
    expect(recorded).not.toContain("acme");
    expect(recorded).not.toContain("Acme Corp");
    expect(recorded).not.toContain("Draft to Bob");
    expect(recorded).not.toContain("hunter2");
    // An unmasked field shows its fixed placeholder, never its value.
    expect(recorded).toContain("Search workflows");
  });

  it("masks the attributes that carry user content", async () => {
    render(
      <div className={MASKED_IN_REPLAY}>
        <span data-text="Review Quarterly report">Review</span>
        <span aria-description="Spend for Maria">Spend</span>
        <div
          role="meter"
          aria-valuenow={12.4}
          aria-valuetext="$12.40 of $50.00 spent"
          aria-roledescription="Maria's meter"
        />
        <select aria-label="Plan">
          <option label="Acme Corp plan" value="acme-plan" />
        </select>
        <button value="Alice Example">Pick</button>
        <a href="/files/1" download="alice-tax-return.pdf">
          Download
        </a>
        <div data-placeholder="Ask Maria anything" />
      </div>,
    );

    const recorded = await recordReplayText();
    for (const userContent of [
      "Quarterly report",
      "$12.40",
      "12.4",
      "Maria",
      "Acme Corp",
      "acme-plan",
      "Alice Example",
      "alice-tax-return",
    ]) {
      expect(recorded).not.toContain(userContent);
    }
  });

  it("shows fixed copy that went through fixedCopy", async () => {
    const label = fixedCopy("Workspace");
    render(
      <Text variant="body" unmask>
        {fixedCopy(label)}
      </Text>,
    );

    expect(await recordReplayText()).toContain("Workspace");
  });
});

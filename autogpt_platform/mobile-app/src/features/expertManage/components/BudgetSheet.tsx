import { useState } from "react";
import { Platform, Text, View } from "react-native";

import { Button } from "@/components/Button";
import { BottomSheet } from "@/features/connections/components/BottomSheet";
import { SheetField } from "@/features/connections/components/CredentialSheet/components/SheetField";
import { createStyles, spacing, type } from "@/lib/theme";

import { budgetInput, parseBudget } from "../helpers";

// iOS's decimal pad shows the region's separator, a comma in much of Europe,
// and the field takes only a point.
const BUDGET_KEYBOARD = Platform.OS === "ios" ? "numbers-and-punctuation" : "decimal-pad";

interface Props {
  isOpen: boolean;
  /** Effective cap in credits. Null means there is no cap. */
  saved: number | null;
  onClose: () => void;
  /** Resolves with an error to show, or null once saved. */
  onSave: (credits: number | null) => Promise<string | null>;
}

export function BudgetSheet({ isOpen, saved, onClose, onSave }: Props) {
  return (
    <BottomSheet isOpen={isOpen} onClose={onClose}>
      {isOpen ? <BudgetForm saved={saved} onSave={onSave} /> : null}
    </BottomSheet>
  );
}

interface FormProps {
  saved: number | null;
  onSave: (credits: number | null) => Promise<string | null>;
}

function BudgetForm({ saved, onSave }: FormProps) {
  const styles = useStyles();
  const [text, setText] = useState(() => budgetInput(saved));
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  function handleChange(next: string) {
    setText(next);
    setError(null);
  }

  async function handleSave() {
    if (isSaving || text === budgetInput(saved)) return;
    const parsed = parseBudget(text);
    if ("error" in parsed) {
      setError(parsed.error);
      return;
    }
    setIsSaving(true);
    const failed = await onSave(parsed.credits);
    setIsSaving(false);
    setError(failed);
  }

  function submit() {
    void handleSave();
  }

  return (
    <View style={styles.body}>
      <Text accessibilityRole="header" style={styles.heading}>
        Weekly budget
      </Text>
      <SheetField
        label="Weekly budget"
        value={text}
        onChangeText={handleChange}
        placeholder="e.g. 25"
        keyboardType={BUDGET_KEYBOARD}
        autoFocus
        onSubmitEditing={submit}
      />
      <Text style={styles.hint}>Leave empty to use the default. 0 turns the limit off.</Text>
      {error ? (
        <Text accessibilityLiveRegion="polite" style={styles.error}>
          {error}
        </Text>
      ) : null}
      <Button
        label="Save"
        onPress={submit}
        isLoading={isSaving}
        disabled={text === budgetInput(saved)}
        fullWidth
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  body: { gap: spacing.md },
  heading: { ...type.headline, color: colors.text },
  hint: { ...type.small, color: colors.textSecondary },
  error: { ...type.small, color: colors.error },
}));

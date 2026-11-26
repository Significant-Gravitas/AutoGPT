import { useState } from "react";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { BottomSheet } from "@/features/connections/components/BottomSheet";
import { SheetField } from "@/features/connections/components/CredentialSheet/components/SheetField";
import type { ExpertSoul } from "@/lib/api/expertManage";
import { createStyles, spacing, type } from "@/lib/theme";

import { isProfileDirty, validateProfile } from "../helpers";

interface Props {
  isOpen: boolean;
  saved: ExpertSoul;
  onClose: () => void;
  /** Resolves with an error to show, or null once saved. */
  onSave: (form: ExpertSoul) => Promise<string | null>;
}

export function EditProfileSheet({ isOpen, saved, onClose, onSave }: Props) {
  return (
    <BottomSheet isOpen={isOpen} onClose={onClose}>
      {isOpen ? <ProfileForm saved={saved} onSave={onSave} /> : null}
    </BottomSheet>
  );
}

interface FormProps {
  saved: ExpertSoul;
  onSave: (form: ExpertSoul) => Promise<string | null>;
}

function ProfileForm({ saved, onSave }: FormProps) {
  const styles = useStyles();
  const [form, setForm] = useState(saved);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  function change(field: keyof ExpertSoul) {
    return (text: string) => {
      setForm((current) => ({ ...current, [field]: text }));
      setError(null);
    };
  }

  async function handleSave() {
    const invalid = validateProfile(form);
    if (invalid) {
      setError(invalid);
      return;
    }
    setIsSaving(true);
    const failed = await onSave(form);
    setIsSaving(false);
    setError(failed);
  }

  function submit() {
    void handleSave();
  }

  return (
    <View style={styles.body}>
      <Text accessibilityRole="header" style={styles.heading}>
        Name and instructions
      </Text>
      <SheetField
        label="Name"
        value={form.name}
        onChangeText={change("name")}
        autoCapitalize="sentences"
      />
      <SheetField
        label="Identity and personality"
        value={form.identity}
        onChangeText={change("identity")}
        autoCapitalize="sentences"
        isMultiline
      />
      <SheetField
        label="Voice"
        value={form.voice_preferences}
        onChangeText={change("voice_preferences")}
        placeholder="How should this expert sound?"
        autoCapitalize="sentences"
        isMultiline
      />
      <SheetField
        label="Boundaries"
        value={form.boundaries}
        onChangeText={change("boundaries")}
        placeholder="What should this expert avoid or ask about first?"
        autoCapitalize="sentences"
        isMultiline
      />
      {error ? (
        <Text accessibilityLiveRegion="polite" style={styles.error}>
          {error}
        </Text>
      ) : null}
      <Button
        label="Save"
        onPress={submit}
        disabled={!isProfileDirty(saved, form)}
        isLoading={isSaving}
        fullWidth
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  body: { gap: spacing.md },
  heading: { ...type.headline, color: colors.text },
  error: { ...type.small, color: colors.error },
}));

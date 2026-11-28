import { useState } from "react";
import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { BottomSheet } from "@/features/connections/components/BottomSheet";
import { SheetField } from "@/features/connections/components/CredentialSheet/components/SheetField";
import { createStyles, spacing, type } from "@/lib/theme";

interface Props {
  isOpen: boolean;
  initialTitle: string;
  onClose: () => void;
  onSubmit: (text: string) => string | null;
}

export function RenameChatSheet({ isOpen, initialTitle, onClose, onSubmit }: Props) {
  return (
    <BottomSheet isOpen={isOpen} onClose={onClose}>
      {isOpen ? <RenameForm initialTitle={initialTitle} onSubmit={onSubmit} /> : null}
    </BottomSheet>
  );
}

interface FormProps {
  initialTitle: string;
  onSubmit: (text: string) => string | null;
}

function RenameForm({ initialTitle, onSubmit }: FormProps) {
  const styles = useStyles();
  const [text, setText] = useState(initialTitle);
  const [error, setError] = useState<string | null>(null);

  function handleChange(next: string) {
    setText(next);
    setError(null);
  }

  function handleSave() {
    setError(onSubmit(text));
  }

  return (
    <View style={styles.body}>
      <Text accessibilityRole="header" style={styles.heading}>
        Rename chat
      </Text>
      <SheetField
        label="Name"
        value={text}
        onChangeText={handleChange}
        autoFocus
        autoCapitalize="sentences"
        onSubmitEditing={handleSave}
      />
      {error ? (
        <Text accessibilityLiveRegion="polite" style={styles.error}>
          {error}
        </Text>
      ) : null}
      <Button label="Save" onPress={handleSave} fullWidth />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  body: { gap: spacing.md },
  heading: { ...type.headline, color: colors.text },
  error: { ...type.small, color: colors.error },
}));

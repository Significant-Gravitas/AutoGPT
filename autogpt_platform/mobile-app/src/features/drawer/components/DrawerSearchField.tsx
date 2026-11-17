import { Search01Icon } from "@hugeicons/core-free-icons";
import { TextInput, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, radius, spacing, type, useColors } from "@/lib/theme";

interface Props {
  value: string;
  onChangeText: (text: string) => void;
}

export function DrawerSearchField({ value, onChangeText }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.field}>
      <Icon icon={Search01Icon} size={18} color={colors.textTertiary} />
      <TextInput
        value={value}
        onChangeText={onChangeText}
        placeholder="Search chats"
        placeholderTextColor={colors.textTertiary}
        accessibilityLabel="Search chats"
        autoCorrect={false}
        autoCapitalize="none"
        returnKeyType="search"
        clearButtonMode="while-editing"
        style={styles.input}
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  field: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    minHeight: 40,
    paddingHorizontal: spacing.md,
    borderRadius: radius.md,
    backgroundColor: colors.surface,
  },
  input: {
    ...type.label,
    fontWeight: "400",
    flex: 1,
    paddingVertical: spacing.sm,
    color: colors.text,
  },
}));

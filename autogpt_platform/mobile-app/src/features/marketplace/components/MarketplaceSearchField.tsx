import { Search01Icon } from "@hugeicons/core-free-icons";
import { TextInput, View } from "react-native";

import { Icon } from "@/components/Icon";
import { createStyles, useColors } from "@/lib/theme";

const PLACEHOLDER = "Search by job or name";

interface Props {
  value: string;
  onChangeText: (text: string) => void;
}

export function MarketplaceSearchField({ value, onChangeText }: Props) {
  const styles = useStyles();
  const colors = useColors();
  return (
    <View style={styles.field}>
      <Icon icon={Search01Icon} size={15} color={colors.textTertiary} />
      <TextInput
        value={value}
        onChangeText={onChangeText}
        placeholder={PLACEHOLDER}
        placeholderTextColor={colors.textTertiary}
        accessibilityLabel={PLACEHOLDER}
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
    gap: 6,
    height: 36,
    marginHorizontal: 14,
    marginBottom: 14,
    paddingHorizontal: 12,
    borderRadius: 11,
    backgroundColor: colors.panelCard,
  },
  input: { flex: 1, height: 36, paddingVertical: 0, fontSize: 13, color: colors.text },
}));

import { Search01Icon } from "@hugeicons/core-free-icons";
import { Pressable, Text, View } from "react-native";

import { IconButton } from "@/components/IconButton";
import { createStyles, spacing, type } from "@/lib/theme";

import { DrawerSearchField } from "./DrawerSearchField";

interface Props {
  isSearchOpen: boolean;
  query: string;
  onChangeQuery: (text: string) => void;
  onOpenSearch: () => void;
  onCloseSearch: () => void;
}

/** The app's name and a search button; searching swaps the name for the field. */
export function DrawerHeader({
  isSearchOpen,
  query,
  onChangeQuery,
  onOpenSearch,
  onCloseSearch,
}: Props) {
  const styles = useStyles();

  if (isSearchOpen) {
    return (
      <View style={[styles.header, styles.searching]}>
        <View style={styles.field}>
          <DrawerSearchField value={query} onChangeText={onChangeQuery} autoFocus />
        </View>
        <Pressable
          accessibilityRole="button"
          hitSlop={8}
          onPress={onCloseSearch}
          style={({ pressed }) => pressed && styles.pressed}
        >
          <Text style={styles.cancel}>Cancel</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.header}>
      <Text accessibilityRole="header" numberOfLines={1} style={styles.title}>
        AutoGPT
      </Text>
      <IconButton
        icon={Search01Icon}
        label="Search chats"
        onPress={onOpenSearch}
        variant="menu"
      />
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.md,
    minHeight: 56,
    paddingLeft: spacing.xxl,
    paddingRight: spacing.xl,
  },
  searching: { paddingLeft: spacing.lg },
  title: {
    ...type.title,
    flex: 1,
    fontSize: 26,
    fontWeight: "700",
    color: colors.text,
  },
  field: { flex: 1 },
  cancel: { ...type.label, fontSize: 16, color: colors.text },
  pressed: { opacity: 0.6 },
}));

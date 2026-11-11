import { Menu01Icon } from "@hugeicons/core-free-icons";
import { useNavigation } from "expo-router";
import { DrawerActions } from "expo-router/react-navigation";
import type { ReactNode } from "react";
import { Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, spacing, type } from "@/lib/theme";

import { IconButton } from "./IconButton";

interface Props {
  title?: string;
  center?: ReactNode;
  right?: ReactNode;
}

export function ScreenHeader({ title, center, right }: Props) {
  const navigation = useNavigation();
  const insets = useSafeAreaInsets();
  const styles = useStyles();

  function openMenu() {
    navigation.dispatch(DrawerActions.openDrawer());
  }

  return (
    <View style={[styles.header, { paddingTop: insets.top + spacing.xs }]}>
      <View style={styles.side}>
        <IconButton icon={Menu01Icon} label="Open menu" onPress={openMenu} />
      </View>
      {center ? (
        <View style={styles.center}>{center}</View>
      ) : (
        <Text numberOfLines={1} style={styles.title}>
          {title}
        </Text>
      )}
      <View style={[styles.side, styles.right]}>{right}</View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
    paddingHorizontal: spacing.sm,
    paddingBottom: spacing.xs,
    backgroundColor: colors.background,
  },
  title: { ...type.heading, flex: 1, color: colors.text, textAlign: "center" },
  center: { flex: 1, alignItems: "center" },
  // Both sides share one width so the title stays centred whatever sits on
  // the right.
  side: { width: 88, flexDirection: "row" },
  right: { justifyContent: "flex-end", gap: spacing.xs },
}));

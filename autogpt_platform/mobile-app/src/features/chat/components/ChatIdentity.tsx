import { StyleSheet, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { colors, spacing, type } from "@/lib/theme";

interface Props {
  name: string;
  role: string | null;
  avatar: string | null;
}

export function ChatIdentity({ name, role, avatar }: Props) {
  return (
    <View accessibilityRole="header" style={styles.row}>
      <ExpertAvatar name={name} source={avatar} size={32} />
      <View style={styles.text}>
        <Text numberOfLines={1} style={styles.name}>
          {name}
        </Text>
        {role ? (
          <Text numberOfLines={1} style={styles.role}>
            {role}
          </Text>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: "row", alignItems: "center", gap: spacing.sm, maxWidth: "100%" },
  text: { flexShrink: 1 },
  name: { ...type.label, color: colors.text },
  role: { ...type.small, fontSize: 12, lineHeight: 16, color: colors.textTertiary },
});

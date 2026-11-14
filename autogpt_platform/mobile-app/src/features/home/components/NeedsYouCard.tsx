import { Text, View } from "react-native";

import { Button } from "@/components/Button";
import { ExpertAvatar } from "@/components/ExpertAvatar";
import { avatarSource } from "@/features/team/helpers";
import type { HomeAttentionItem } from "@/lib/api/home";
import { expertTint } from "@/lib/expertPalette";
import { createStyles, radius, spacing } from "@/lib/theme";

import { openHomeLink } from "../openHomeLink";

interface Props {
  items: HomeAttentionItem[];
}

export function NeedsYouCard({ items }: Props) {
  const styles = useStyles();
  const [item] = items;
  if (!item) return null;
  const { expert } = item;

  function handlePress() {
    openHomeLink(item.primary_action.href);
  }

  return (
    <View style={styles.card}>
      <View style={styles.top}>
        <Text style={styles.eyebrow}>NEEDS YOU</Text>
        {items.length > 1 ? (
          <Text style={styles.count}>{`${items.length} waiting`}</Text>
        ) : null}
      </View>
      <View style={styles.row}>
        {expert ? (
          <ExpertAvatar
            name={expert.name}
            source={avatarSource(expert)}
            size={40}
            tint={expertTint({ avatarUrl: expert.avatar_url, role: expert.role })}
          />
        ) : null}
        <View style={styles.text}>
          <Text numberOfLines={1} style={styles.title}>
            {item.title}
          </Text>
          <Text numberOfLines={2} style={styles.description}>
            {item.description}
          </Text>
        </View>
        <Button
          label={item.primary_action.label}
          onPress={handlePress}
          variant="primary"
          size="small"
        />
      </View>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    gap: spacing.sm,
    padding: spacing.lg,
    borderRadius: radius.lg,
    backgroundColor: colors.brandSoft,
    borderWidth: 0.5,
    // `brand` at about 30% opacity.
    borderColor: `${colors.brand}4D`,
  },
  top: { flexDirection: "row", justifyContent: "space-between" },
  eyebrow: {
    fontSize: 12,
    fontWeight: "600",
    letterSpacing: 0.5,
    color: colors.accentText,
  },
  count: { fontSize: 12, color: colors.textTertiary },
  row: { flexDirection: "row", alignItems: "center", gap: spacing.md },
  text: { flex: 1, gap: 2 },
  title: { fontSize: 15, fontWeight: "500", lineHeight: 20, color: colors.text },
  description: { fontSize: 13, lineHeight: 18, color: colors.textSecondary },
}));

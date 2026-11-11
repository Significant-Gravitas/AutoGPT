import { Pressable, ScrollView, Text, View } from "react-native";

import { Icon } from "@/components/Icon";
import { promptTopic } from "@/features/chat/promptTopic";
import { mixHex } from "@/lib/expertPalette";
import {
  createStyles,
  radius,
  spacing,
  type,
  useColors,
  useScheme,
} from "@/lib/theme";

const CARD_WIDTH = 168;
const TILE_SIZE = 36;

// How much of the topic colour goes into each surface, per scheme: dark
// surfaces need less of a pastel to read as tinted.
const MIX = {
  dark: { card: 7, border: 22, tile: 20, icon: 75 },
  light: { card: 12, border: 30, tile: 30, icon: 60 },
} as const;

interface PromptExpert {
  name: string;
  tint: string;
}

interface Props {
  prompts: string[];
  /** Set when the chat goes to an expert: every card wears their colour and name. */
  expert?: PromptExpert | null;
  onPick: (prompt: string) => void;
}

export function PromptChips({ prompts, expert, onPick }: Props) {
  const styles = useStyles();
  return (
    <ScrollView
      horizontal
      keyboardShouldPersistTaps="handled"
      showsHorizontalScrollIndicator={false}
      // A horizontal ScrollView grows to fill its parent unless told not to.
      style={styles.scroller}
      contentContainerStyle={styles.row}
    >
      {prompts.map((prompt) => (
        <PromptCard key={prompt} prompt={prompt} expert={expert} onPick={onPick} />
      ))}
    </ScrollView>
  );
}

interface CardProps {
  prompt: string;
  expert?: PromptExpert | null;
  onPick: (prompt: string) => void;
}

function PromptCard({ prompt, expert, onPick }: CardProps) {
  const styles = useStyles();
  const colors = useColors();
  const mix = MIX[useScheme()];
  const topic = promptTopic(prompt);
  const tint = expert?.tint ?? topic.tint;
  const accent = mixHex(tint, mix.icon, colors.text);

  function handlePress() {
    onPick(prompt);
  }

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={prompt}
      onPress={handlePress}
      style={({ pressed }) => [
        styles.card,
        {
          backgroundColor: mixHex(tint, mix.card, colors.surface),
          borderColor: mixHex(tint, mix.border, colors.border),
        },
        pressed && styles.pressed,
      ]}
    >
      <View style={styles.top}>
        <View
          style={[styles.tile, { backgroundColor: mixHex(tint, mix.tile, colors.surface) }]}
        >
          <Icon icon={topic.icon} size={18} color={accent} />
        </View>
        {expert ? (
          <Text numberOfLines={1} style={[styles.name, { color: accent }]}>
            {expert.name}
          </Text>
        ) : null}
      </View>
      <Text numberOfLines={3} style={styles.label}>
        {prompt}
      </Text>
    </Pressable>
  );
}

const useStyles = createStyles((colors) => ({
  scroller: { flexGrow: 0 },
  row: { gap: spacing.sm, paddingHorizontal: spacing.lg },
  card: {
    width: CARD_WIDTH,
    gap: spacing.md,
    padding: spacing.md,
    borderRadius: radius.lg,
    borderWidth: 1,
  },
  pressed: { opacity: 0.7, transform: [{ scale: 0.97 }] },
  top: { flexDirection: "row", alignItems: "center", gap: spacing.sm },
  name: { ...type.small, fontWeight: "600", flexShrink: 1 },
  tile: {
    width: TILE_SIZE,
    height: TILE_SIZE,
    borderRadius: radius.md,
    alignItems: "center",
    justifyContent: "center",
  },
  label: { ...type.small, fontWeight: "500", color: colors.text },
}));

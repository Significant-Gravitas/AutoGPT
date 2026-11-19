import { Add01Icon } from "@hugeicons/core-free-icons";
import { ActivityIndicator, Pressable, ScrollView, Text, View } from "react-native";

import { ExpertAvatar } from "@/components/ExpertAvatar";
import { Icon } from "@/components/Icon";
import { avatarSource, avatarTint } from "@/features/team/helpers";
import type { Expert } from "@/lib/api/experts";
import { createStyles, radius, useColors, useScheme } from "@/lib/theme";

import { areaColor, seatTint } from "../areaColors";
import { type Area, firstName, type OpenSeat } from "../helpers";

const SEAT_SIZE = 40;

interface Props {
  team: Expert[];
  openSeats: OpenSeat[];
  openingID: string | null;
  onOpenExpert: (expert: Expert) => void;
  onOpenSeat: (area: Area) => void;
}

/** Who is on the team, then one open seat per area nobody covers. */
export function TeamCard({ team, openSeats, openingID, onOpenExpert, onOpenSeat }: Props) {
  const styles = useStyles();
  const colors = useColors();
  const scheme = useScheme();
  return (
    <View style={styles.card}>
      <View style={styles.heading}>
        <Text accessibilityRole="header" style={styles.title}>
          Your team
        </Text>
        <Text style={styles.count}>{`${team.length} hired`}</Text>
      </View>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.seats}
      >
        {team.map((expert) => (
          <Pressable
            key={expert.id}
            accessibilityRole="button"
            accessibilityLabel={`Chat with ${expert.name}`}
            accessibilityState={{ busy: openingID === expert.id }}
            onPress={() => onOpenExpert(expert)}
            style={({ pressed }) => [styles.seat, pressed && styles.pressed]}
          >
            <View>
              <ExpertAvatar
                name={expert.name}
                source={avatarSource(expert)}
                tint={avatarTint(expert)}
                size={SEAT_SIZE}
              />
              {openingID === expert.id ? (
                <View style={styles.opening}>
                  <ActivityIndicator size="small" color={colors.text} />
                </View>
              ) : null}
            </View>
            <Text numberOfLines={1} style={styles.name}>
              {firstName(expert.name)}
            </Text>
          </Pressable>
        ))}
        {openSeats.map((seat) => (
          <Pressable
            key={seat.area}
            accessibilityRole="button"
            accessibilityLabel={`${seat.label}: nobody on your team yet`}
            accessibilityHint={`Shows the ${seat.label} experts`}
            onPress={() => onOpenSeat(seat.area)}
            style={({ pressed }) => [styles.seat, pressed && styles.pressed]}
          >
            <View
              style={[
                styles.dashed,
                {
                  borderColor: areaColor(seat.area),
                  backgroundColor: seatTint(seat.area, scheme),
                },
              ]}
            >
              <Icon icon={Add01Icon} size={18} color={areaColor(seat.area)} />
            </View>
            <Text numberOfLines={1} style={styles.seatLabel}>
              {seat.label}
            </Text>
          </Pressable>
        ))}
      </ScrollView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  card: {
    marginHorizontal: 14,
    marginBottom: 18,
    paddingTop: 12,
    paddingBottom: 10,
    borderRadius: radius.lg,
    backgroundColor: colors.panelCard,
    overflow: "hidden",
  },
  heading: {
    flexDirection: "row",
    alignItems: "baseline",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingBottom: 10,
  },
  title: { fontSize: 13.5, lineHeight: 18, fontWeight: "600", color: colors.text },
  count: { fontSize: 11, lineHeight: 15, color: colors.textTertiary },
  seats: { gap: 10, paddingHorizontal: 12 },
  seat: { alignItems: "center", gap: 5, minWidth: SEAT_SIZE, maxWidth: 64 },
  pressed: { opacity: 0.6 },
  opening: {
    position: "absolute",
    top: 0,
    left: 0,
    width: SEAT_SIZE,
    height: SEAT_SIZE,
    borderRadius: SEAT_SIZE / 2,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.pageWash,
  },
  name: { fontSize: 10, lineHeight: 13, color: colors.text },
  seatLabel: { fontSize: 10, lineHeight: 13, color: colors.textTertiary },
  dashed: {
    width: SEAT_SIZE,
    height: SEAT_SIZE,
    borderRadius: SEAT_SIZE / 2,
    borderWidth: 1.5,
    borderStyle: "dashed",
    alignItems: "center",
    justifyContent: "center",
  },
}));

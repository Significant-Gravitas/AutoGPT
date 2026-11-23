import { FlatList, ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { createStyles, spacing } from "@/lib/theme";

import { type TableData, TABLE_ROW_LIMIT } from "../../libraryHelpers";

const NUMBER_WIDTH = 40;

interface Props {
  table: TableData;
  /** Set when only the start of the file came down. */
  cappedNotice: string | null;
}

/** A CSV or TSV as a table: headings that stay put, row numbers, sideways scrolling. */
export function TableView({ table, cappedNotice }: Props) {
  const styles = useStyles();
  const insets = useSafeAreaInsets();
  const width = NUMBER_WIDTH + table.widths.reduce((sum, column) => sum + column, 0);
  const shown =
    table.totalRows > TABLE_ROW_LIMIT
      ? `First ${TABLE_ROW_LIMIT} of ${table.totalRows} rows`
      : `${table.totalRows} ${table.totalRows === 1 ? "row" : "rows"}`;

  function renderCells(cells: string[], isHeading: boolean) {
    return table.widths.map((columnWidth, column) => (
      <Text
        key={column}
        numberOfLines={isHeading ? 1 : 2}
        style={[
          styles.cell,
          isHeading && styles.heading,
          { width: columnWidth },
          table.numeric[column] && styles.number,
        ]}
      >
        {cells[column] ?? ""}
      </Text>
    ));
  }

  return (
    <View style={styles.screen}>
      <Text style={styles.note}>
        {`${shown} · ${table.widths.length} ${table.widths.length === 1 ? "column" : "columns"}`}
        {cappedNotice ? ` · ${cappedNotice}` : ""}
      </Text>
      <ScrollView horizontal contentContainerStyle={styles.horizontal}>
        <FlatList
          style={{ width }}
          data={table.rows}
          keyExtractor={(_, index) => String(index)}
          stickyHeaderIndices={[0]}
          contentContainerStyle={{ paddingBottom: insets.bottom + spacing.xl }}
          ListHeaderComponent={
            <View style={[styles.row, styles.headingRow]}>
              <Text style={[styles.cell, styles.rowNumber, styles.heading]} />
              {renderCells(table.header, true)}
            </View>
          }
          renderItem={({ item, index }) => (
            <View style={styles.row}>
              <Text style={[styles.cell, styles.rowNumber]}>{index + 1}</Text>
              {renderCells(item, false)}
            </View>
          )}
        />
      </ScrollView>
    </View>
  );
}

const useStyles = createStyles((colors) => ({
  screen: { flex: 1 },
  note: { fontSize: 12, lineHeight: 16, color: colors.textTertiary, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm },
  horizontal: { paddingHorizontal: spacing.sm },
  row: {
    flexDirection: "row",
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
    backgroundColor: colors.background,
  },
  headingRow: { backgroundColor: colors.surface },
  cell: { fontSize: 13, lineHeight: 18, color: colors.text, paddingHorizontal: 10, paddingVertical: 8 },
  heading: { fontWeight: "600", color: colors.textSecondary },
  number: { textAlign: "right", fontVariant: ["tabular-nums"] },
  rowNumber: { width: NUMBER_WIDTH, textAlign: "right", color: colors.textTertiary, paddingHorizontal: 8 },
}));

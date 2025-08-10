import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/views/task/task_list_tile.dart';
import 'package:auto_gpt_flutter_client/models/task.dart';

void main() {
  final Task testTask = Task(id: 1, title: "Sample Task");

  testWidgets('TaskListTile displays the task title',
      (WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(
        home: TaskListTile(task: testTask, onTap: () {}, onDelete: () {})));
    expect(find.text('Sample Task'), findsOneWidget);
  });

  testWidgets('TaskListTile toggles isSelected state on tap',
      (WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(
        home: TaskListTile(task: testTask, onTap: () {}, onDelete: () {})));

    // Initially, the delete icon should not be present
    expect(find.byIcon(Icons.close), findsNothing);

    // Tap the tile
    await tester.tap(find.text('Sample Task'));
    await tester.pump();

    // The delete icon should appear
    expect(find.byIcon(Icons.close), findsOneWidget);
  });

  testWidgets('TaskListTile triggers onDelete when delete icon is tapped',
      (WidgetTester tester) async {
    bool wasDeleteCalled = false;
    await tester.pumpWidget(MaterialApp(
        home: TaskListTile(
            task: testTask,
            onTap: () {},
            onDelete: () {
              wasDeleteCalled = true;
            })));

    // Tap the tile to make the delete icon appear
    await tester.tap(find.text('Sample Task'));
    await tester.pump();

    // Tap the delete icon
    await tester.tap(find.byIcon(Icons.close));
    await tester.pump();

    expect(wasDeleteCalled, true);
  });

  testWidgets('TaskListTile triggers onTap when tapped',
      (WidgetTester tester) async {
    bool wasTapped = false;
    await tester.pumpWidget(MaterialApp(
        home: TaskListTile(
            task: testTask,
            onTap: () {
              wasTapped = true;
            },
            onDelete: () {})));

    // Tap the tile
    await tester.tap(find.text('Sample Task'));
    await tester.pump();

    expect(wasTapped, true);
  });
}

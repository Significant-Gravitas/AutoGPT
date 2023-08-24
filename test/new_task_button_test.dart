import 'package:auto_gpt_flutter_client/views/task/new_task_button.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('NewTaskButton triggers callback on press',
      (WidgetTester tester) async {
    bool wasPressed = false;

    // Build our widget.
    await tester.pumpWidget(MaterialApp(
      home: Scaffold(
        body: NewTaskButton(onPressed: () => wasPressed = true),
      ),
    ));

    // Verify if the button with the text 'New Task' is displayed.
    expect(find.text('New Task'), findsOneWidget);

    // Tap the button and verify if the onPressed callback is triggered.
    await tester.tap(find.byType(ElevatedButton));
    expect(wasPressed, true);
  });
}

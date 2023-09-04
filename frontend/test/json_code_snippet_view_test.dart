import 'package:auto_gpt_flutter_client/views/chat/json_code_snippet_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const jsonString = '{"key": "value"}';

  testWidgets('Renders JsonCodeSnippetView without crashing',
      (WidgetTester tester) async {
    await tester.pumpWidget(
        const MaterialApp(home: JsonCodeSnippetView(jsonString: jsonString)));
    expect(find.byType(JsonCodeSnippetView), findsOneWidget);
  });
}

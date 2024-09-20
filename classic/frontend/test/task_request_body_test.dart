import 'package:auto_gpt_flutter_client/models/task_request_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('TaskRequestBody', () {
    test('should create TaskRequestBody with correct values', () {
      final taskRequestBody = TaskRequestBody(
          input: 'Do something', additionalInput: {'key': 'value'});

      expect(taskRequestBody.input, 'Do something');
      expect(taskRequestBody.additionalInput, {'key': 'value'});
    });

    test('should convert TaskRequestBody to correct JSON', () {
      final taskRequestBody = TaskRequestBody(
          input: 'Do something', additionalInput: {'key': 'value'});

      final json = taskRequestBody.toJson();

      expect(json, {
        'input': 'Do something',
        'additional_input': {'key': 'value'}
      });
    });
  });
}

import 'package:auto_gpt_flutter_client/models/step_request_body.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('StepRequestBody', () {
    test('should create StepRequestBody with correct values', () {
      final stepRequestBody = StepRequestBody(
          input: 'Execute something', additionalInput: {'key': 'value'});

      expect(stepRequestBody.input, 'Execute something');
      expect(stepRequestBody.additionalInput, {'key': 'value'});
    });

    test('should convert StepRequestBody to correct JSON', () {
      final stepRequestBody = StepRequestBody(
          input: 'Execute something', additionalInput: {'key': 'value'});

      final json = stepRequestBody.toJson();

      expect(json, {
        'input': 'Execute something',
        'additional_input': {'key': 'value'}
      });
    });
  });
}

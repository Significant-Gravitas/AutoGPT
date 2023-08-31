class StepRequestBody {
  final String input;
  final Map<String, dynamic>? additionalInput;

  StepRequestBody({required this.input, this.additionalInput});

  Map<String, dynamic> toJson() {
    return {'input': input, 'additional_input': additionalInput};
  }
}

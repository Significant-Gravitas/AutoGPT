class StepRequestBody {
  final String? input;
  final Map<String, dynamic>? additionalInput;

  StepRequestBody({required this.input, this.additionalInput});

  Map<String, dynamic> toJson() {
    if (input == null && additionalInput == null) {
      return {};
    }
    return {'input': input, 'additional_input': additionalInput};
  }
}

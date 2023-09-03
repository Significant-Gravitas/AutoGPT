import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:auto_gpt_flutter_client/viewmodels/api_settings_viewmodel.dart';

class ApiBaseUrlField extends StatelessWidget {
  final TextEditingController controller;

  const ApiBaseUrlField({required this.controller});

  @override
  Widget build(BuildContext context) {
    return Consumer<ApiSettingsViewModel>(
      builder: (context, apiSettingsViewModel, child) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Column(
            children: [
              Container(
                height: 50,
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: Colors.black, width: 0.5),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  child: TextField(
                    controller: controller,
                    decoration: const InputDecoration(
                      border: InputBorder.none,
                      hintText: 'API Base URL',
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton(
                    onPressed: () {
                      controller.text = 'http://127.0.0.1:8000';
                      apiSettingsViewModel.updateBaseURL(controller.text);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: Colors.black,
                      textStyle: const TextStyle(
                        color: Colors.black,
                      ),
                    ),
                    child: const Text("Reset"),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      apiSettingsViewModel.updateBaseURL(controller.text);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: Colors.black,
                      textStyle: const TextStyle(
                        color: Colors.black,
                      ),
                    ),
                    child: const Text("Update"),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }
}

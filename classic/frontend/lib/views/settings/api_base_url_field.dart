import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class ApiBaseUrlField extends StatelessWidget {
  final TextEditingController controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Consumer<SettingsViewModel>(
      builder: (context, settingsViewModel, child) {
        // TODO: This view shouldn't know about the settings view model. It should use a delegate
        controller.text = settingsViewModel.baseURL;
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
                      hintText: 'Agent Base URL',
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
                      controller.text = 'http://127.0.0.1:8000/ap/v1';
                      settingsViewModel.updateBaseURL(controller.text);
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
                      settingsViewModel.updateBaseURL(controller.text);
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

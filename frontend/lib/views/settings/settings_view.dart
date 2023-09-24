import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:flutter/material.dart';

/// [SettingsView] displays a list of settings that the user can configure.
/// It uses [SettingsViewModel] for state management and logic.
class SettingsView extends StatelessWidget {
  final SettingsViewModel viewModel;

  /// Constructor for [SettingsView], requiring an instance of [SettingsViewModel].
  const SettingsView({Key? key, required this.viewModel}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.grey,
        foregroundColor: Colors.black,
        title: const Text('Settings'),
      ),
      body: ListView(
        children: [
          // Dark Mode Toggle
          SwitchListTile(
            title: const Text('Dark Mode'),
            value: viewModel.isDarkModeEnabled,
            onChanged: viewModel.toggleDarkMode,
          ),
          // Developer Mode Toggle
          SwitchListTile(
            title: const Text('Developer Mode'),
            value: viewModel.isDeveloperModeEnabled,
            onChanged: viewModel.toggleDeveloperMode,
          ),
          // Base URL Configuration
          ListTile(
            title: const Text('Base URL'),
            subtitle: TextFormField(
              initialValue: viewModel.baseURL,
              onChanged: viewModel.updateBaseURL,
              decoration: const InputDecoration(
                hintText: 'Enter Base URL',
              ),
            ),
          ),
          // Continuous Mode Steps Configuration
          ListTile(
            title: const Text('Continuous Mode Steps'),
            // User can increment or decrement the number of steps using '+' and '-' buttons.
            subtitle: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.remove),
                  onPressed: viewModel
                      .decrementContinuousModeSteps, // Decrement the number of steps.
                ),
                Text('${viewModel.continuousModeSteps} Steps'),
                IconButton(
                  icon: const Icon(Icons.add),
                  onPressed: viewModel
                      .incrementContinuousModeSteps, // Increment the number of steps.
                ),
              ],
            ),
          ),
          ListTile(
            title: Text('Sign Out'),
            onTap: () {
              viewModel.signOut();
            },
          ),
        ],
      ),
    );
  }
}

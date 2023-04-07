import subprocess

def is_package_installed(package):
    try:
        if package in ["npx", "npm"]:
            output = subprocess.check_output(f"npm list -g --depth=0 {package}", shell=True, text=True)
        else:
            output = subprocess.check_output(f"npm list --depth=0 {package}", shell=True, text=True)
        
        if package in output:
            return True
        else:
            return False
    except subprocess.CalledProcessError:
        return False

def install_packages(packages):
    if not isinstance(packages, list):
        return "Invalid input. Please provide a list of packages to be installed."

    installed_packages = []
    for package in packages:
        if is_package_installed(package):
            continue

        if package in ["npx", "npm"]:
            command = f"sudo npm install -g {package}"
        else:
            command = f"sudo npm install {package}"

        try:
            subprocess.run(command, shell=True, check=True, text=True)
            installed_packages.append(package)
        except subprocess.CalledProcessError as e:
            return f"An error occurred during the installation of {package}: {e}"

    return f"Successfully installed {', '.join(installed_packages)}." if installed_packages else "All packages are already installed."

def create_react_app(app_name):
    try:
        command = f"npx create-react-app {app_name}"
        subprocess.run(command, shell=True, check=True, text=True)
        return f"Successfully created a new React app named '{app_name}'."
    except subprocess.CalledProcessError as e:
        return f"An error occurred during the creation of the React app: {e}"

def run_react_app(app_name):
    try:
        command = f"cd {app_name} && npm start"
        subprocess.run(command, shell=True, check=True, text=True)
        return f"Successfully started the development server for '{app_name}'."
    except subprocess.CalledProcessError as e:
        return f"An error occurred while starting the development server: {e}"

def build_react_app(app_name):
    try:
        command = f"cd {app_name} && npm run build"
        subprocess.run(command, shell=True, check=True, text=True)
        return f"Successfully built the React app '{app_name}' for production."
    except subprocess.CalledProcessError as e:
        return f"An error occurred while building the React app: {e}"

def run_react_app_tests(app_name):
    try:
        command = f"cd {app_name} && npm test"
        subprocess.run(command, shell=True, check=True, text=True)
        return f"Successfully ran tests for the React app '{app_name}'."
    except subprocess.CalledProcessError as e:
        return f"An error occurred while running tests for the React app: {e}"

def eject_react_app(app_name):
    try:
        command = f"cd {app_name} && npm run eject"
        subprocess.run(command, shell=True, check=True, text=True)
        return f"Successfully ejected the React app '{app_name}' from create-react-app."
    except subprocess.CalledProcessError as e:
        return f"An error occurred while ejecting the React app: {e}"

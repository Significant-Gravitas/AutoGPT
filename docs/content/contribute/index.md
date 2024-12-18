# 🚀 Contributing to the Docs

Welcome to our documentation contribution guide! We're excited to have you here.  
This guide will help you get started with contributing to our documentation.  
**Let's make our docs better together! 💪**

---

## 🛠️ Setting up the Docs

### Prerequisites

- Git installed on your machine
- Python 3.8 or higher
- A text editor of your choice

---

### Step-by-Step Setup Guide

1. **Clone the Repository** 📥  
   ```shell
   git clone https://github.com/Significant-Gravitas/AutoGPT.git
   ```

2. **Install Dependencies** 📚  
   ```shell
   python -m pip install -r docs/requirements.txt
   ```
   **or**  
   ```shell
   python3 -m pip install -r docs/requirements.txt
   ```

3. **Start the Development Server** 🔥  
   ```shell
   mkdocs serve
   ```
   - Open your browser and navigate to:  
     `http://127.0.0.1:8000`
   - Use `-a localhost:8392` to run on a different port.

4. **View the Docs** 🌐  
   - Changes will auto-reload in your browser.

---

## 📝 Adding New Content

### Creating a New Page

1. **Create the File** 📄  
   - Navigate to `docs/content/`  
   - Create a new markdown file.

2. **Update Navigation** 🗺️  
   - Open `mkdocs.yml`  
   - Add the new file to the `nav` section.

3. **Verify the Page** ✅  
   - Run the development server to check changes.  

---

## 🔍 Quality Checks

1. Test all links before submitting.
2. Follow existing documentation style.
3. Use proper Markdown formatting.
4. Include code examples where helpful.

---

## 🎯 Submitting Your Contribution

### Pull Request Process

1. **Create a Branch** 🌿  
   - Use a descriptive name for your branch.  
   - Keep changes focused and atomic.

2. **Submit PR** 📋  
   - Fill out the PR template.  
   - Add screenshots if applicable.  
   - Link related issues.

3. **Review Process** 🧵  
   - Address reviewer feedback.  
   - Make requested changes.  
   - Maintain active communication.

---

## 🌟 Best Practices

- Keep documentation clear and concise.
- Test all links before submitting.
- Follow existing documentation style.
- Include helpful code examples.

---

**Thank you for contributing to our documentation! ❤️ Together, we make documentation better for everyone.**

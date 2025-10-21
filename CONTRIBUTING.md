# Contributing to ChronoPlay

Thank you for your interest in contributing to ChronoPlay! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/chronoplay/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Detailed description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages or logs

### Suggesting Enhancements

1. Check if the enhancement has already been suggested
2. Create a new issue with:
   - Clear description of the enhancement
   - Use cases and benefits
   - Possible implementation approach

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style (PEP 8 for Python)
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**
   ```bash
   # Run basic tests
   python generation/generation.py --help
   python evaluation/retrieval_runner.py --help
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Explain what testing was done

## 📝 Code Style Guidelines

### Python Code Style

- Follow PEP 8
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

Example:
```python
def generate_qa_pair(template: Dict[str, Any], docs: List[Dict]) -> Dict[str, Any]:
    """
    Generate a QA pair from a template and retrieved documents.
    
    Args:
        template: Question template with placeholders
        docs: Retrieved context documents
        
    Returns:
        Generated QA pair dictionary
    """
    # Implementation
    pass
```

### Documentation

- Update README.md for new features
- Add inline comments for complex logic
- Include usage examples for new functions

### Commit Messages

Use clear, descriptive commit messages:
- `Add: New feature description`
- `Fix: Bug description`
- `Update: What was updated`
- `Docs: Documentation changes`

## 🧪 Testing

Before submitting:

1. **Test basic functionality**
   ```bash
   # Test generation
   python generation/generation.py --game_name dyinglight2 --segment_id 1 --help
   
   # Test retrieval
   python evaluation/retrieval_runner.py --game dyinglight2 --segment_id 1 --help
   ```

2. **Check for common issues**
   - No hardcoded paths
   - No exposed API keys
   - Proper error handling
   - Clear error messages

## 🎯 Areas for Contribution

We welcome contributions in:

1. **New Features**
   - Additional retrieval methods
   - New evaluation metrics
   - Support for more games
   - Enhanced QA filtering strategies

2. **Improvements**
   - Performance optimizations
   - Better error handling
   - Enhanced logging
   - Code refactoring

3. **Documentation**
   - Tutorials and examples
   - API documentation
   - Translation to other languages

4. **Data**
   - New game knowledge
   - Question templates
   - User personas

## 📋 Development Setup

1. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/chronoplay.git
   cd chronoplay
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**
   ```bash
   # Create .env file (not committed to git)
   echo "OPENAI_API_KEY=your-key-here" > .env
   echo "OPENAI_BASE_URL=your-url-here" >> .env
   ```

## 🔍 Code Review Process

1. Maintainers review all pull requests
2. Feedback provided within a few days
3. Make requested changes if any
4. Once approved, PR will be merged

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 💬 Questions?

Feel free to:
- Open an issue for questions
- Start a discussion in GitHub Discussions
- Contact maintainers directly

## 🙏 Thank You!

Your contributions help make ChronoPlay better for everyone!


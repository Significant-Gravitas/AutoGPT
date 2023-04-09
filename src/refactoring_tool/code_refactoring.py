from typing import List, Optional

from .code_analysis import CodeAnalysis
from .code_improvement import CodeImprovement
from .test_generation import TestGeneration


class RefactoringOptions:
    def __init__(
        self,
        analyze_readability: bool = True,
        analyze_performance: bool = True,
        analyze_security: bool = False,
        analyze_modularity: bool = True,
        refactor_variables: bool = True,
        optimize_loops: bool = True,
        simplify_conditionals: bool = True,
        refactor_functions: bool = True,
        generate_unit_tests: bool = True,
        generate_integration_tests: bool = False,
        generate_system_tests: bool = False,
    ):
        self.analyze_readability = analyze_readability
        self.analyze_performance = analyze_performance
        self.analyze_security = analyze_security
        self.analyze_modularity = analyze_modularity
        self.refactor_variables = refactor_variables
        self.optimize_loops = optimize_loops
        self.simplify_conditionals = simplify_conditionals
        self.refactor_functions = refactor_functions
        self.generate_unit_tests = generate_unit_tests
        self.generate_integration_tests = generate_integration_tests
        self.generate_system_tests = generate_system_tests


def refactor_code(code: str, options: Optional[RefactoringOptions] = None) -> str:
    if options is None:
        options = RefactoringOptions()

    # Analyze code
    analysis_results = []
    if options.analyze_readability:
        analysis_result = CodeAnalysis.analyze_code_readability(code)
        if analysis_result:  # Make sure the result is not empty
            analysis_results.append(analysis_result)
    if options.analyze_performance:
        analysis_results.extend(CodeAnalysis.analyze_code_performance(code))
    if options.analyze_security:
        analysis_results.extend(CodeAnalysis.analyze_code_security(code))
    if options.analyze_modularity:
        analysis_results.extend(CodeAnalysis.analyze_code_modularity(code))

    # Improve code
    if options.refactor_variables:
        code = CodeImprovement.refactor_variables(analysis_results, code)
    if options.optimize_loops:
        code = CodeImprovement.optimize_loops(analysis_results, code)
    if options.simplify_conditionals:
        code = CodeImprovement.simplify_conditionals(analysis_results, code)
    if options.refactor_functions:
        code = CodeImprovement.refactor_functions(analysis_results, code)

    # Generate tests
    test_code = ""
    if options.generate_unit_tests:
        test_code += TestGeneration.write_tests(code)
    if options.generate_integration_tests:
        test_code += TestGeneration.write_integration_tests(code)
    if options.generate_system_tests:
        test_code += TestGeneration.write_system_tests(code)

    return code, test_code

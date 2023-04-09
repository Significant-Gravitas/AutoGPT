from typing import List, Optional
from dataclasses import dataclass

from .code_analysis import (
    analyze_code_readability,
    analyze_code_performance,
    analyze_code_security,
    analyze_code_modularity,
)
from .code_improvement import (
    refactor_variables,
    optimize_loops,
    simplify_conditionals,
    refactor_functions,
)
from .test_generation import (
    write_tests,
    write_integration_tests,
    write_system_tests,
)

@dataclass
class RefactoringOptions:
    analyze_readability: bool = True
    analyze_performance: bool = True
    analyze_security: bool = False
    analyze_modularity: bool = True
    refactor_variables: bool = True
    optimize_loops: bool = True
    simplify_conditionals: bool = True
    refactor_functions: bool = True
    generate_unit_tests: bool = True
    generate_integration_tests: bool = False
    generate_system_tests: bool = False

def refactor_code(code: str, options: Optional[RefactoringOptions] = None) -> str:
    if options is None:
        options = RefactoringOptions()

    # Analyze code
    analysis_results = []
    if options.analyze_readability:
        analysis_results.extend(analyze_code_readability(code))
    if options.analyze_performance:
        analysis_results.extend(analyze_code_performance(code))
    if options.analyze_security:
        analysis_results.extend(analyze_code_security(code))
    if options.analyze_modularity:
        analysis_results.extend(analyze_code_modularity(code))

    # Improve code
    if options.refactor_variables:
        code = refactor_variables(analysis_results, code)
    if options.optimize_loops:
        code = optimize_loops(analysis_results, code)
    if options.simplify_conditionals:
        code = simplify_conditionals(analysis_results, code)
    if options.refactor_functions:
        code = refactor_functions(analysis_results, code)

    # Generate tests
    test_code = ""
    if options.generate_unit_tests:
        test_code += write_tests(code)
    if options.generate_integration_tests:
        test_code += write_integration_tests(code)
    if options.generate_system_tests:
        test_code += write_system_tests(code)

    return code, test_code

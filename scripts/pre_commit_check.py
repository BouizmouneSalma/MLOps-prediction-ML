"""
Script de vérification pré-commit
Exécute toutes les vérifications avant de committer
"""
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print a header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def run_check(name, command, critical=True):
    """Run a check command"""
    print(f"\n{Colors.BOLD}Running {name}...{Colors.END}")
    result = subprocess.run(
        command, 
        shell=True, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print_success(f"{name} passed")
        return True
    else:
        if critical:
            print_error(f"{name} failed")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        else:
            print_warning(f"{name} failed (non-critical)")
        return not critical


def main():
    """Main pre-commit check function"""
    print_header("PRE-COMMIT CHECKS")
    
    checks_passed = []
    checks_failed = []
    
    # Check 1: Black formatting
    if run_check("Black Code Formatting", "black --check .", critical=False):
        checks_passed.append("Black")
    else:
        checks_failed.append("Black")
        print_warning("Run 'black .' to format code")
    
    # Check 2: Flake8 linting
    if run_check(
        "Flake8 Linting", 
        "flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics",
        critical=True
    ):
        checks_passed.append("Flake8")
    else:
        checks_failed.append("Flake8")
    
    # Check 3: Data validation
    if Path("data/data.csv").exists():
        if run_check("Data Validation", "python scripts/validate_data.py", critical=True):
            checks_passed.append("Data Validation")
        else:
            checks_failed.append("Data Validation")
    else:
        print_warning("Data file not found, skipping data validation")
    
    # Check 4: API tests
    if Path("tests/test_api.py").exists():
        if run_check("API Tests", "pytest tests/test_api.py -v", critical=False):
            checks_passed.append("API Tests")
        else:
            checks_failed.append("API Tests")
    else:
        print_warning("Test file not found, skipping API tests")
    
    # Print summary
    print_header("SUMMARY")
    
    print(f"\n{Colors.BOLD}Passed Checks ({len(checks_passed)}):{Colors.END}")
    for check in checks_passed:
        print_success(check)
    
    if checks_failed:
        print(f"\n{Colors.BOLD}Failed Checks ({len(checks_failed)}):{Colors.END}")
        for check in checks_failed:
            print_error(check)
    
    # Final result
    print("\n" + "=" * 60)
    if not checks_failed:
        print_success("All critical checks passed! Ready to commit.")
        print("=" * 60 + "\n")
        return 0
    else:
        print_error("Some checks failed. Please fix the issues before committing.")
        print("=" * 60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Script de validation des données pour le pipeline CI/CD
Vérifie la qualité et l'intégrité des données avant l'entraînement
"""
import sys
import json
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_validation import validate_dataset


def main():
    """Main validation function"""
    print("=" * 60)
    print("DATA VALIDATION PIPELINE")
    print("=" * 60)
    
    # Load dataset
    data_path = Path(__file__).parent.parent / "data" / "data.csv"
    
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        sys.exit(1)
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Run validation
    print("\n Running validation checks...")
    is_valid, validation_report = validate_dataset(df, target_column="Cluster")
    
    # Create reports directory if it doesn't exist
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Save validation report
    report_path = reports_dir / "data_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"\n Validation report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for check_name, check_result in validation_report.items():
        if isinstance(check_result, dict):
            print(f"\n{check_name.upper()}:")
            for key, value in check_result.items():
                print(f"  - {key}: {value}")
        else:
            status = "✓" if check_result else "❌"
            print(f"{status} {check_name}: {check_result}")
    
    # Exit with appropriate code
    if is_valid:
        print("\nAll validation checks passed!")
        sys.exit(0)
    else:
        print("\n Validation failed! Please review the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

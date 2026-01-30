"""
Script de vérification des performances du modèle
Vérifie que le modèle atteint les seuils de performance minimaux
"""
import sys
import json
from pathlib import Path


# Seuils de performance minimaux
PERFORMANCE_THRESHOLDS = {
    "accuracy": 0.70,    # 70% accuracy minimum
    "f1_score": 0.65,    # 65% F1 score minimum
    "precision": 0.60,   # 60% precision minimum
    "recall": 0.60       # 60% recall minimum
}


def main():
    """Check model performance against thresholds"""
    print("=" * 60)
    print("MODEL PERFORMANCE CHECK")
    print("=" * 60)
    
    # Load metrics report
    report_path = Path(__file__).parent.parent / "reports" / "model_metrics.json"
    
    if not report_path.exists():
        print(f"❌ ERROR: Metrics report not found at {report_path}")
        sys.exit(1)
    
    with open(report_path, "r") as f:
        report = json.load(f)
    
    metrics = report.get("metrics", {})
    best_model = report.get("best_model", "Unknown")
    
    print(f"\n🤖 Best Model: {best_model}")
    print("\n📊 Performance Check:")
    
    all_passed = True
    failed_checks = []
    
    for metric_name, threshold in PERFORMANCE_THRESHOLDS.items():
        actual_value = metrics.get(metric_name, 0)
        passed = actual_value >= threshold
        status = "✓" if passed else "❌"
        
        print(f"  {status} {metric_name}: {actual_value:.4f} (threshold: {threshold:.4f})")
        
        if not passed:
            all_passed = False
            failed_checks.append({
                "metric": metric_name,
                "actual": actual_value,
                "threshold": threshold,
                "difference": threshold - actual_value
            })
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ All performance checks passed!")
        print("Model meets minimum performance requirements.")
        sys.exit(0)
    else:
        print("❌ Performance check failed!")
        print("\nFailed checks:")
        for check in failed_checks:
            print(f"  - {check['metric']}: {check['actual']:.4f} < {check['threshold']:.4f} "
                  f"(missing {check['difference']:.4f})")
        print("\nModel does not meet minimum performance requirements.")
        sys.exit(1)


if __name__ == "__main__":
    main()

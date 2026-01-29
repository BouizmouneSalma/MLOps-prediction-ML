import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any


class DataValidator:
    
    def __init__(self, target_column: str = "Cluster"):
        self.target_column = target_column
        self.validation_report = {}
        
    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        checks = [
            self._check_target_exists(df),
            self._check_missing_values(df),
            self._check_data_types(df),
            self._check_infinite_values(df)
        ]
        
        is_valid = all(check[0] for check in checks)
        
        self.validation_report = {
            "is_valid": is_valid,
            "total_samples": len(df),
            "n_features": len(df.columns) - 1 if self.target_column in df.columns else len(df.columns),
            "checks": {
                "target_exists": checks[0][1],
                "missing_values": checks[1][1],
                "data_types": checks[2][1],
                "infinite_values": checks[3][1]
            }
        }
        
        return is_valid, self.validation_report
    
    def _check_target_exists(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        exists = self.target_column in df.columns
        
        result = {
            "target_column": self.target_column,
            "exists": exists,
            "unique_values": int(df[self.target_column].nunique()) if exists else 0
        }
        
        return exists, result
    
    def _check_missing_values(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        result = {
            "has_missing": bool(total_missing > 0),
            "total_missing": int(total_missing)
        }
        
        return total_missing == 0, result
    
    def _check_data_types(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        X = df.drop(self.target_column, axis=1) if self.target_column in df.columns else df
        
        non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        
        result = {
            "all_numeric": len(non_numeric_cols) == 0,
            "non_numeric_columns": non_numeric_cols
        }
        
        return len(non_numeric_cols) == 0, result
    
    def _check_infinite_values(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        X = df.drop(self.target_column, axis=1) if self.target_column in df.columns else df
        
        total_inf = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        
        result = {
            "has_infinite": bool(total_inf > 0),
            "total_infinite": int(total_inf)
        }
        
        return total_inf == 0, result
    
    # display data report (missing values , datatypes ect)
    def print_report(self):
        """Print validation report"""
        if not self.validation_report:
            print("No validation report available.")
            return

        print(f"Status: {'✓ PASSED' if self.validation_report['is_valid'] else '✗ FAILED'}")
        print(f"Samples: {self.validation_report['total_samples']}")
        print(f"Features: {self.validation_report['n_features']}")
        print("-"*70)
        
        checks = self.validation_report['checks']
        
        # Target
        if checks['target_exists']['exists']:
            print(f"✓ Target '{checks['target_exists']['target_column']}' found ({checks['target_exists']['unique_values']} classes)")
        else:
            print(f"✗ Target column not found")
        
        if checks['missing_values']['has_missing']:
            print(f"✗ Missing values: {checks['missing_values']['total_missing']}")
        else:
            print("✓ No missing values")
        
        if checks['data_types']['all_numeric']:
            print("✓ All features numeric")
        else:
            print(f"✗ Non-numeric: {checks['data_types']['non_numeric_columns']}")
        
        if checks['infinite_values']['has_infinite']:
            print(f"✗ Infinite values: {checks['infinite_values']['total_infinite']}")
        else:
            print("✓ No infinite values")
        

def validate_dataset(df: pd.DataFrame, target_column: str = "Cluster") -> Tuple[bool, Dict]:

    validator = DataValidator(target_column=target_column)
    is_valid, report = validator.validate_all(df)
    validator.print_report()
    
    return is_valid, report

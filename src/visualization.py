"""
Visualization module for employment projections model analysis.

This module provides comprehensive visualization functions for model evaluation,
including prediction analysis, feature importance plots, and performance metrics.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


def set_style():
    """Set consistent plotting style."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10


def plot_actual_vs_predicted(y_true, y_pred, save_path=None):
    """
    Create scatter plot of actual vs predicted values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        save_path: Optional path to save the plot
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate and display metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Add metrics text box
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            verticalalignment='top', fontsize=12)
    
    ax.set_xlabel('Actual Employment 2034 (thousands)', fontsize=12)
    ax.set_ylabel('Predicted Employment 2034 (thousands)', fontsize=12)
    ax.set_title('Actual vs Predicted Employment Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Actual vs Predicted plot saved to {save_path}")
    
    plt.close()  # Close instead of show


def plot_residuals(y_true, y_pred, save_path=None):
    """
    Create residuals plot to identify patterns in prediction errors.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        save_path: Optional path to save the plot
    """
    set_style()
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax2.text(0.05, 0.95, f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plot saved to {save_path}")
    
    plt.close()


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """
    Create horizontal bar plot of feature importances.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Optional path to save the plot
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature importances.")
        return
    
    set_style()
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=True).tail(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                   color=sns.color_palette("viridis", len(importance_df)))
    
    # Customize plot
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.close()


def plot_prediction_distribution(y_true, y_pred, save_path=None):
    """
    Plot distribution of actual vs predicted values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        save_path: Optional path to save the plot
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histograms
    ax.hist(y_true, bins=30, alpha=0.7, label='Actual Values', color='skyblue', edgecolor='black')
    ax.hist(y_pred, bins=30, alpha=0.7, label='Predicted Values', color='orange', edgecolor='black')
    
    # Add vertical lines for means
    ax.axvline(np.mean(y_true), color='blue', linestyle='--', linewidth=2, label=f'Actual Mean: {np.mean(y_true):.2f}')
    ax.axvline(np.mean(y_pred), color='red', linestyle='--', linewidth=2, label=f'Predicted Mean: {np.mean(y_pred):.2f}')
    
    ax.set_xlabel('Employment 2034 (thousands)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution: Actual vs Predicted Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    plt.close()


def plot_prediction_errors_by_magnitude(y_true, y_pred, save_path=None):
    """
    Analyze prediction errors by magnitude of actual values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        save_path: Optional path to save the plot
    """
    set_style()
    
    # Calculate absolute and percentage errors
    abs_errors = np.abs(y_true - y_pred)
    pct_errors = (abs_errors / y_true) * 100
    
    # Create bins based on actual values
    bins = np.percentile(y_true, [0, 25, 50, 75, 100])
    bin_labels = ['Q1 (Small)', 'Q2 (Medium-Low)', 'Q3 (Medium-High)', 'Q4 (Large)']
    
    # Assign each prediction to a bin
    bin_assignments = np.digitize(y_true, bins) - 1
    bin_assignments = np.clip(bin_assignments, 0, len(bin_labels) - 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot of absolute errors by magnitude
    abs_error_data = [abs_errors[bin_assignments == i] for i in range(len(bin_labels))]
    box1 = ax1.boxplot(abs_error_data, labels=bin_labels, patch_artist=True)
    
    for patch in box1['boxes']:
        patch.set_facecolor('lightblue')
    
    ax1.set_xlabel('Employment Magnitude Quartiles', fontsize=12)
    ax1.set_ylabel('Absolute Error', fontsize=12)
    ax1.set_title('Prediction Errors by Employment Magnitude', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Box plot of percentage errors by magnitude
    pct_error_data = [pct_errors[bin_assignments == i] for i in range(len(bin_labels))]
    box2 = ax2.boxplot(pct_error_data, labels=bin_labels, patch_artist=True)
    
    for patch in box2['boxes']:
        patch.set_facecolor('lightcoral')
    
    ax2.set_xlabel('Employment Magnitude Quartiles', fontsize=12)
    ax2.set_ylabel('Percentage Error (%)', fontsize=12)
    ax2.set_title('Percentage Errors by Employment Magnitude', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis plot saved to {save_path}")
    
    plt.close()


def create_model_summary_report():
    """
    Create a comprehensive model summary with all visualizations.
    """
    print("Creating comprehensive model evaluation report...")
    
    # Load model and data
    try:
        model = joblib.load('model_data/random_forest_model.joblib')
        selected_features = pd.read_csv('model_data/selected_features.csv')
        target = pd.read_csv('model_data/target.csv').squeeze()
        predictions_df = pd.read_csv('model_data/predictions.csv')
        
        # Extract predictions
        y_pred = predictions_df['Predicted Employment 2034'].values
        y_true = target.values
        
        # Get feature names
        feature_names = selected_features.columns.tolist()
        
        print(f"Loaded {len(y_true)} samples for evaluation")
        print(f"Model has {len(feature_names)} features")
        
        # Create plots directory
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate all visualizations
        print("\n1. Creating Actual vs Predicted plot...")
        plot_actual_vs_predicted(y_true, y_pred, 
                                os.path.join(plots_dir, 'actual_vs_predicted.png'))
        
        print("\n2. Creating Residuals analysis...")
        plot_residuals(y_true, y_pred, 
                      os.path.join(plots_dir, 'residuals_analysis.png'))
        
        print("\n3. Creating Feature Importance plot...")
        plot_feature_importance(model, feature_names, 
                               save_path=os.path.join(plots_dir, 'feature_importance.png'))
        
        print("\n4. Creating Distribution comparison...")
        plot_prediction_distribution(y_true, y_pred, 
                                   os.path.join(plots_dir, 'distribution_comparison.png'))
        
        print("\n5. Creating Error analysis by magnitude...")
        plot_prediction_errors_by_magnitude(y_true, y_pred, 
                                          os.path.join(plots_dir, 'error_analysis.png'))
        
        # Print summary statistics
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
        print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
        print(f"Mean Actual: {np.mean(y_true):.2f}")
        print(f"Mean Predicted: {np.mean(y_pred):.2f}")
        print(f"Std Actual: {np.std(y_true):.2f}")
        print(f"Std Predicted: {np.std(y_pred):.2f}")
        print("="*60)
        print(f"All plots saved to '{plots_dir}/' directory")
        
        return {
            'model': model,
            'y_true': y_true,
            'y_pred': y_pred,
            'feature_names': feature_names
        }
        
    except Exception as e:
        print(f"Error creating model summary: {e}")
        raise


if __name__ == "__main__":
    create_model_summary_report()
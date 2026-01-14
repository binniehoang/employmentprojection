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
    ax.scatter(y_true, y_pred, alpha=0.6, s=50)
    
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
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
    # Use small epsilon for zero actual values to avoid division by zero
    epsilon = np.finfo(float).eps
    safe_y_true = np.where(y_true == 0, epsilon, y_true)
    pct_errors = (abs_errors / safe_y_true) * 100
    
    
    # Create bins based on actual values
    bins = np.percentile(y_true, [0, 25, 50, 75, 100])
    bin_labels = ['Q1 (Small)', 'Q2 (Medium-Low)', 'Q3 (Medium-High)', 'Q4 (Large)']
    
    # Assign each prediction to a bin
    bin_assignments = np.digitize(y_true, bins) - 1
    bin_assignments = np.clip(bin_assignments, 0, len(bin_labels) - 1)
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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


def plot_temporal_employment_progression(save_path=None):
    """
    Create visualization showing employment progression from 2024 to 2034.
    Shows before/after comparison and growth trends.
    
    Args:
        save_path: Optional path to save the plot
    """
    set_style()
    
    try:
        # Load data with both 2024 and 2034 employment
        predictions_df = pd.read_csv('model_data/predictions.csv')
        
        employment_2024 = predictions_df['Employment 2024'].values
        predicted_2034 = predictions_df['Predicted Employment 2034'].values
        
        # Calculate change and percent change
        employment_change = predicted_2034 - employment_2024
        percent_change = ((predicted_2034 - employment_2024) / employment_2024) * 100
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Before vs After Comparison
        ax1.scatter(employment_2024, predicted_2034, alpha=0.6, s=50, c='steelblue')
        
        # Add diagonal line for no change
        min_val = min(employment_2024.min(), predicted_2034.min())
        max_val = max(employment_2024.max(), predicted_2034.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8, label='No Change Line')
        
        ax1.set_xlabel('Employment 2024 (thousands)', fontsize=12)
        ax1.set_ylabel('Predicted Employment 2034 (thousands)', fontsize=12)
        ax1.set_title('Employment Progression: 2024 → 2034', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Employment Change Distribution
        colors = ['red' if x < 0 else 'green' for x in employment_change]
        ax2.hist(employment_change, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
        ax2.axvline(x=np.mean(employment_change), color='orange', linestyle='-', linewidth=2, 
                   label=f'Mean Change: {np.mean(employment_change):.1f}k')
        
        ax2.set_xlabel('Employment Change 2024-2034 (thousands)', fontsize=12)
        ax2.set_ylabel('Number of Occupations', fontsize=12)
        ax2.set_title('Distribution of Employment Changes (2024-2034)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Percent Change Distribution
        ax3.hist(percent_change, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
        ax3.axvline(x=np.mean(percent_change), color='darkred', linestyle='-', linewidth=2, 
                   label=f'Mean % Change: {np.mean(percent_change):.1f}%')
        
        ax3.set_xlabel('Employment Percent Change 2024-2034 (%)', fontsize=12)
        ax3.set_ylabel('Number of Occupations', fontsize=12)
        ax3.set_title('Distribution of Employment Percent Changes (2024-2034)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Growth vs Decline Analysis
        growing_occupations = employment_change > 0
        declining_occupations = employment_change < 0
        stable_occupations = employment_change == 0
        
        categories = ['Growing\n(+)', 'Declining\n(-)', 'Stable\n(0)']
        counts = [
            np.sum(growing_occupations),
            np.sum(declining_occupations), 
            np.sum(stable_occupations)
        ]
        colors_pie = ['lightgreen', 'lightcoral', 'lightgray']
        
        wedges, texts, autotexts = ax4.pie(counts, labels=categories, colors=colors_pie, 
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title('Occupation Growth Categories (2024-2034)', fontsize=14, fontweight='bold')
        
        # Add count labels
        for i, (count, autotext) in enumerate(zip(counts, autotexts)):
            autotext.set_text(f'{count}\n({count/sum(counts)*100:.1f}%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temporal progression plot saved to {save_path}")
        
        plt.close()
        
        # Print summary statistics
        print(f"\n📊 TEMPORAL ANALYSIS SUMMARY (2024-2034)")
        print(f"{'='*50}")
        print(f"Total Occupations Analyzed: {len(employment_change):,}")
        print(f"Growing Occupations: {np.sum(growing_occupations):,} ({np.sum(growing_occupations)/len(employment_change)*100:.1f}%)")
        print(f"Declining Occupations: {np.sum(declining_occupations):,} ({np.sum(declining_occupations)/len(employment_change)*100:.1f}%)")
        print(f"Average Employment Change: {np.mean(employment_change):.1f}k jobs")
        print(f"Average Percent Change: {np.mean(percent_change):.1f}%")
        print(f"Largest Growth: {np.max(employment_change):.1f}k jobs")
        print(f"Largest Decline: {np.min(employment_change):.1f}k jobs")
        
    except Exception as e:
        print(f"Error creating temporal progression plot: {e}")


def plot_occupation_timeline_sample(top_n=20, save_path=None):
    """
    Create timeline visualization for top N occupations showing 2024 vs 2034 employment.
    
    Args:
        top_n: Number of top occupations by 2034 employment to show
        save_path: Optional path to save the plot
    """
    set_style()
    
    try:
        # Load data
        predictions_df = pd.read_csv('model_data/predictions.csv')
        raw_data = pd.read_csv('data/cleaned_employment_projections.csv')
        
        # Merge to get occupation titles
        if 'Occupation Title' in raw_data.columns:
            merged_data = predictions_df.merge(raw_data[['Occupation Title']], 
                                             left_index=True, right_index=True, how='left')
        else:
            # If no occupation titles, create generic ones
            merged_data = predictions_df.copy()
            merged_data['Occupation Title'] = [f'Occupation {i+1}' for i in range(len(predictions_df))]
        
        # Get top N occupations by 2034 employment
        top_occupations = merged_data.nlargest(top_n, 'Predicted Employment 2034')
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create timeline plot
        y_positions = range(len(top_occupations))
        
        # Plot lines connecting 2024 to 2034
        for i, (_, row) in enumerate(top_occupations.iterrows()):
            emp_2024 = row['Employment 2024']
            emp_2034 = row['Predicted Employment 2034']
            
            # Color based on growth/decline
            color = 'green' if emp_2034 > emp_2024 else 'red' if emp_2034 < emp_2024 else 'gray'
            
            # Plot line
            ax.plot([emp_2024, emp_2034], [i, i], color=color, linewidth=2, alpha=0.7)
            
            # Plot points
            ax.scatter([emp_2024], [i], color='blue', s=60, alpha=0.8, label='2024' if i == 0 else "")
            ax.scatter([emp_2034], [i], color='orange', s=60, alpha=0.8, label='2034 (Predicted)' if i == 0 else "")
        
        # Customize plot
        occupation_labels = [title[:40] + '...' if len(title) > 40 else title 
                           for title in top_occupations['Occupation Title']]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(occupation_labels, fontsize=9)
        ax.set_xlabel('Employment (thousands)', fontsize=12)
        ax.set_title(f'Employment Timeline: Top {top_n} Occupations (2024 → 2034)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add annotation
        ax.text(0.02, 0.98, 'Green: Growing | Red: Declining | Gray: Stable', 
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Occupation timeline plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error creating occupation timeline plot: {e}")


def plot_growth_sectors_analysis(save_path=None):
    """
    Analyze and visualize which sectors/wage levels show the most growth.
    
    Args:
        save_path: Optional path to save the plot
    """
    set_style()
    
    try:
        predictions_df = pd.read_csv('model_data/predictions.csv')
        
        employment_2024 = predictions_df['Employment 2024'].values
        predicted_2034 = predictions_df['Predicted Employment 2034'].values
        wages = predictions_df['Median Annual Wage 2024'].values
        
        # Calculate changes
        employment_change = predicted_2034 - employment_2024
        percent_change = ((predicted_2034 - employment_2024) / employment_2024) * 100
        
        # Create wage categories
        wage_bins = np.percentile(wages[wages > 0], [0, 33, 66, 100])  # Exclude 0 wages
        wage_labels = ['Low Wage\n(Bottom 33%)', 'Medium Wage\n(Middle 33%)', 'High Wage\n(Top 33%)']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Employment Change vs Wage Level
        mask = wages > 0  # Only include occupations with wage data
        ax1.scatter(wages[mask], employment_change[mask], alpha=0.6, s=50, c=employment_change[mask], 
                   cmap='RdYlGn', vmin=-50, vmax=50)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Median Annual Wage 2024 ($)', fontsize=12)
        ax1.set_ylabel('Employment Change 2024-2034 (thousands)', fontsize=12)
        ax1.set_title('Employment Change vs Wage Level', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Average change by wage category
        wage_categories = np.digitize(wages, wage_bins) - 1
        wage_categories = np.clip(wage_categories, 0, len(wage_labels) - 1)
        
        avg_changes = []
        for i in range(len(wage_labels)):
            category_mask = (wage_categories == i) & (wages > 0)
            if np.any(category_mask):
                avg_changes.append(np.mean(employment_change[category_mask]))
            else:
                avg_changes.append(0)
        
        colors = ['red' if x < 0 else 'green' for x in avg_changes]
        bars = ax2.bar(wage_labels, avg_changes, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax2.set_ylabel('Average Employment Change (thousands)', fontsize=12)
        ax2.set_title('Average Employment Change by Wage Category', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
                    f'{value:.1f}k', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        # 3. Employment size vs growth rate
        ax3.scatter(employment_2024, percent_change, alpha=0.6, s=50, c=percent_change, 
                   cmap='RdYlGn', vmin=-20, vmax=20)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Employment 2024 (thousands)', fontsize=12)
        ax3.set_ylabel('Employment Percent Change (%)', fontsize=12)
        ax3.set_title('Growth Rate vs Current Employment Size', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Growth distribution by employment size
        # Create size categories
        size_bins = np.percentile(employment_2024, [0, 50, 90, 100])
        size_labels = ['Small\n(Bottom 50%)', 'Medium\n(50-90%)', 'Large\n(Top 10%)']
        size_categories = np.digitize(employment_2024, size_bins) - 1
        size_categories = np.clip(size_categories, 0, len(size_labels) - 1)
        
        # Box plot of growth rates by size
        growth_by_size = [percent_change[size_categories == i] for i in range(len(size_labels))]
        box_plot = ax4.boxplot(growth_by_size, labels=size_labels, patch_artist=True)
        
        colors_box = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors_box):
            patch.set_facecolor(color)
        
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax4.set_ylabel('Employment Percent Change (%)', fontsize=12)
        ax4.set_title('Growth Rate Distribution by Employment Size', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Growth sectors analysis plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error creating growth sectors analysis: {e}")


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
        
        print("\n6. Creating Temporal Employment Progression (2024→2034)...")
        plot_temporal_employment_progression(
            save_path=os.path.join(plots_dir, 'employment_progression_2024_2034.png'))
        
        print("\n7. Creating Occupation Timeline for Top Employers...")
        plot_occupation_timeline_sample(top_n=20, 
                                      save_path=os.path.join(plots_dir, 'top_occupations_timeline.png'))
        
        print("\n8. Creating Growth Sectors Analysis...")
        plot_growth_sectors_analysis(
            save_path=os.path.join(plots_dir, 'growth_sectors_analysis.png'))
        
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
"""
Script to generate temporal employment projection visualizations (2024-2034)
that clearly show the time progression and changes over the decade.
"""

from src.visualization import (
    plot_temporal_employment_progression,
    plot_occupation_timeline_sample, 
    plot_growth_sectors_analysis
)
import os

def main():
    """Generate temporal plots that show 2024-2034 progression."""
    print("Generating temporal employment projection visualizations...")
    
    # Ensure plots directory exists
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n1. Creating Employment Progression Overview (2024→2034)...")
    plot_temporal_employment_progression(
        save_path=os.path.join(plots_dir, 'employment_progression_2024_2034.png'))
    
    print("\n2. Creating Timeline for Top 20 Occupations...")
    plot_occupation_timeline_sample(
        top_n=20, 
        save_path=os.path.join(plots_dir, 'top_occupations_timeline.png'))
    
    print("\n3. Creating Growth Sectors Analysis by Wage Level...")
    plot_growth_sectors_analysis(
        save_path=os.path.join(plots_dir, 'growth_sectors_analysis.png'))
    
    print(f"\n✅ All temporal plots saved to '{plots_dir}/' directory")
    print("\nThese plots now clearly show:")
    print("📈 Employment changes from 2024 to 2034")
    print("📊 Which occupations are growing vs declining")
    print("💰 How wage levels relate to employment growth")
    print("⏱️ Timeline visualization of employment progression")

if __name__ == "__main__":
    main()
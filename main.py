"""
Main execution script for RL-based What-If Analysis Agent
"""
import json
import argparse
import os
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from src.models.data_models import Project
from src.optimization.scenario_generator import ScenarioGenerator
from src.optimization.pareto_optimizer import ParetoOptimizer


def print_scenario_summary(title: str, metrics: Dict[str, Any]):
    """Print formatted scenario summary"""
    print(f"\n  [{title}]")
    print(f"    Duration: {metrics['total_time_days']:.1f} days")
    print(f"    Cost: ${metrics['total_cost']:,.2f}")
    print(f"    Quality: {metrics['quality_score']:.2%}")
    print(f"    Constraints Met: {'Yes' if metrics.get('constraints_satisfied', True) else 'No'}")


def print_allocation_details(scenario: Dict[str, Any], project_data: Dict):
    """Print detailed resource-to-task allocation"""
    print("\n    Resource Allocation Details:")
    print("    " + "-" * 70)
    print(f"    {'Task':<30} {'Resource':<20} {'Hours':<10} {'Cost':<10}")
    print("    " + "-" * 70)
    
    # Create lookup dictionaries
    tasks = {t['id']: t['name'] for t in project_data['tasks']}
    resources = {r['id']: r for r in project_data['resources']}
    
    # Get assignments from scenario
    assignments = scenario.get('assignments', [])
    
    if not assignments:
        print("    No assignments found in this scenario")
        print("    " + "-" * 70)
        return
    
    total_cost = 0
    total_hours = 0
    for assignment in assignments:
        # Handle both dict and object formats
        if isinstance(assignment, dict):
            task_id = assignment.get('task_id')
            resource_id = assignment.get('resource_id')
            hours = assignment.get('hours_allocated', 0)
        else:
            task_id = getattr(assignment, 'task_id', None)
            resource_id = getattr(assignment, 'resource_id', None)
            hours = getattr(assignment, 'hours_allocated', 0)
        
        task_name = tasks.get(task_id, task_id or 'Unknown')
        resource = resources.get(resource_id, {})
        resource_name = resource.get('name', resource_id or 'Unknown')
        cost = hours * resource.get('hourly_rate', 0)
        total_cost += cost
        total_hours += hours
        
        # Truncate long names for display
        task_display = task_name[:28] + '..' if len(task_name) > 30 else task_name
        resource_display = resource_name[:18] + '..' if len(resource_name) > 20 else resource_name
        
        print(f"    {task_display:<30} {resource_display:<20} {hours:<10.1f} ${cost:<9,.0f}")
    
    print("    " + "-" * 70)
    print(f"    {'TOTAL':<30} {'':<20} {total_hours:<10.1f} ${total_cost:<9,.0f}")
    total_days = total_hours / 8  # Convert hours to work days
    print(f"    Total Work Days: {total_days:.1f} days | Total Cost: ${total_cost:,.0f}")


def run_analysis(json_file_path: str, include_rl: bool = True, visualize: bool = False):
    """
    Run complete what-if analysis on project data
    
    Args:
        json_file_path: Path to JSON file with project data
        include_rl: Whether to include RL-optimized scenarios
        visualize: Whether to generate visualization plots
    """
    print("\n" + "="*80)
    print("  RL-BASED WHAT-IF ANALYSIS AGENT FOR PROCESS OPTIMIZATION")
    print("="*80)
    
    # Load project data
    print(f"\n[Loading] Project data from: {json_file_path}")
    project = Project.from_json_file(json_file_path)
    
    # Load raw JSON data for allocation details
    import json
    with open(json_file_path, 'r') as f:
        project_data = json.load(f)
    
    print(f"\n[Project Overview]")
    print(f"  - Name: {project.name}")
    print(f"  - Tasks: {len(project.tasks)}")
    print(f"  - Resources: {len(project.resources)}")
    print(f"  - Estimated Budget: ${project.metadata.estimated_budget:,.2f}")
    
    # Generate scenarios
    print("\n[Generating What-If Scenarios]")
    generator = ScenarioGenerator(project)
    
    print("  > Generating baseline scenario...")
    print("  > Generating parallel execution scenario...")
    print("  > Generating cost-optimized scenario...")
    print("  > Generating balanced scenario...")
    
    if include_rl:
        print("  > Training RL agents for optimization...")
    
    scenarios = generator.generate_all_scenarios(include_rl=include_rl)
    print(f"\n[Success] Generated {len(scenarios)} scenarios")
    
    # Evaluate scenarios
    print("\n[Evaluating] Scenarios with Pareto Optimization...")
    optimizer = ParetoOptimizer(project)
    
    # Find Pareto frontier
    pareto_scenarios = optimizer.find_pareto_frontier(scenarios)
    print(f"  - Found {len(pareto_scenarios)} Pareto-optimal scenarios")
    
    # Generate report
    report = optimizer.generate_report(scenarios)
    
    # Display results
    print("\n" + "="*80)
    print("  OPTIMIZATION RESULTS")
    print("="*80)
    
    print("\n*** BEST SCENARIOS BY OBJECTIVE ***")
    
    # Best Time
    if report['best_scenarios']['fastest']:
        print_scenario_summary(
            "FASTEST COMPLETION",
            report['best_scenarios']['fastest']['metrics']
        )
        print_allocation_details(
            report['best_scenarios']['fastest']['scenario'],
            project_data
        )
    
    # Best Cost
    if report['best_scenarios']['cheapest']:
        print_scenario_summary(
            "LOWEST COST",
            report['best_scenarios']['cheapest']['metrics']
        )
        print_allocation_details(
            report['best_scenarios']['cheapest']['scenario'],
            project_data
        )
    
    # Best Quality
    if report['best_scenarios']['highest_quality']:
        print_scenario_summary(
            "HIGHEST QUALITY",
            report['best_scenarios']['highest_quality']['metrics']
        )
        print_allocation_details(
            report['best_scenarios']['highest_quality']['scenario'],
            project_data
        )
    
    # Best Balanced
    if report['best_scenarios']['best_balanced']:
        print_scenario_summary(
            "BEST BALANCED",
            report['best_scenarios']['best_balanced']['metrics']
        )
        print_allocation_details(
            report['best_scenarios']['best_balanced']['scenario'],
            project_data
        )
    
    # Pareto Frontier
    print("\n" + "="*80)
    print("  PARETO FRONTIER (Non-dominated Solutions)")
    print("="*80)
    
    for i, pareto_data in enumerate(report['pareto_frontier'], 1):
        metrics = pareto_data['metrics']
        print(f"\n  {i}. {pareto_data['scenario']['name']}")
        print(f"     Time: {metrics['total_time_days']:.1f} days | "
              f"Cost: ${metrics['total_cost']:,.2f} | "
              f"Quality: {metrics['quality_score']:.2%}")
    
    # Rankings
    print("\n" + "="*80)
    print("  ALL SCENARIOS RANKED BY OVERALL SCORE")
    print("="*80)
    
    for scenario_data in report['all_scenarios_ranked'][:5]:  # Top 5
        metrics = scenario_data['metrics']
        print(f"\n  Rank #{scenario_data['rank']}: {scenario_data['scenario']['name']}")
        print(f"     Score: {metrics['overall_score']:.3f} | "
              f"Time: {metrics['total_time_days']:.1f}d | "
              f"Cost: ${metrics['total_cost']:,.2f} | "
              f"Quality: {metrics['quality_score']:.2%}")
    
    # Recommendations
    print("\n" + "="*80)
    print("  RECOMMENDATIONS")
    print("="*80)
    
    best_balanced = report['best_scenarios']['best_balanced']
    if best_balanced:
        print(f"\n[RECOMMENDED SCENARIO] {best_balanced['scenario']['name']}")
        print("\nRationale:")
        metrics = best_balanced['metrics']
        
        # Check if it's Pareto optimal
        is_pareto = any(
            p['scenario']['id'] == best_balanced['scenario']['id'] 
            for p in report['pareto_frontier']
        )
        
        if is_pareto:
            print("  - This scenario is Pareto-optimal (non-dominated)")
        
        print(f"  - Achieves project completion in {metrics['total_time_days']:.1f} days")
        print(f"  - Total cost of ${metrics['total_cost']:,.2f} "
              f"({'within' if metrics['total_cost'] <= project.metadata.estimated_budget else 'exceeds'} budget)")
        print(f"  - High quality score of {metrics['quality_score']:.2%}")
        print(f"  - Efficient resource utilization at {metrics['resource_utilization']:.2%}")
        
        if metrics['constraint_violations'] == 0:
            print("  - Satisfies all project constraints")
        else:
            print(f"  - Has {metrics['constraint_violations']} constraint violation(s)")
    
    # Save report
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[Saved] Full report to: {report_file}")
    
    # Visualize if requested
    if visualize:
        print("\n[Generating] Visualization...")
        viz_file = output_dir / "pareto_frontier.png"
        optimizer.visualize_pareto_frontier(scenarios, save_path=str(viz_file))
        print(f"   Visualization saved to: {viz_file}")
    
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print("="*80)
    
    return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RL-based What-If Analysis Agent for Process Optimization"
    )
    parser.add_argument(
        "json_file",
        nargs='?',
        default="example/hospital_project.json",
        help="Path to JSON file with project data (default: example/hospital_project.json)"
    )
    parser.add_argument(
        "--no-rl",
        action="store_true",
        help="Skip RL-optimized scenarios (faster execution)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.json_file):
        print(f"[Error] File '{args.json_file}' not found!")
        return
    
    # Run analysis
    try:
        run_analysis(
            args.json_file,
            include_rl=not args.no_rl,
            visualize=args.visualize
        )
    except Exception as e:
        print(f"\n[Error] During analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

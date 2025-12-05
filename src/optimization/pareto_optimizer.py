"""
Pareto Optimization and Evaluation Metrics for Scenarios
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.data_models import Scenario, Project


@dataclass
class ScenarioMetrics:
    """Comprehensive metrics for scenario evaluation"""
    scenario_id: str
    total_time_days: float
    total_cost: float
    quality_score: float
    resource_utilization: float
    parallelization_factor: float
    skill_match_score: float
    constraint_violations: int
    overall_score: float
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'scenario_id': self.scenario_id,
            'total_time_days': round(self.total_time_days, 2),
            'total_cost': round(self.total_cost, 2),
            'quality_score': round(self.quality_score, 3),
            'resource_utilization': round(self.resource_utilization, 3),
            'parallelization_factor': round(self.parallelization_factor, 3),
            'skill_match_score': round(self.skill_match_score, 3),
            'constraint_violations': self.constraint_violations,
            'overall_score': round(self.overall_score, 3)
        }


class ParetoOptimizer:
    """Pareto optimization for multi-objective scenario evaluation"""
    
    def __init__(self, project: Project):
        """Initialize Pareto optimizer"""
        self.project = project
        self.pareto_frontier = []
        self.all_metrics = []
    
    def evaluate_scenario(self, scenario: Scenario) -> ScenarioMetrics:
        """Calculate comprehensive metrics for a scenario"""
        # Time metrics
        total_time_days = scenario.total_duration_hours / 8
        
        # Cost metrics
        total_cost = scenario.total_cost
        
        # Quality metrics
        quality_score = scenario.quality_score
        
        # Resource utilization
        resource_utilization = self._calculate_resource_utilization(scenario)
        
        # Parallelization factor
        parallelization_factor = self._calculate_parallelization(scenario)
        
        # Skill match score
        skill_match_score = self._calculate_skill_match(scenario)
        
        # Constraint violations
        constraint_violations = self._count_constraint_violations(scenario)
        
        # Overall score (weighted combination)
        overall_score = self._calculate_overall_score(
            total_time_days,
            total_cost,
            quality_score,
            resource_utilization,
            parallelization_factor,
            constraint_violations
        )
        
        scenario_data = {
            'id': scenario.id,
            'name': scenario.name,
            'total_duration_days': scenario.total_duration_hours / 8,
            'total_cost': scenario.total_cost,
            'quality_score': scenario.quality_score,
            'constraints_satisfied': scenario.constraints_satisfied,
            'optimization_type': scenario.optimization_type,
            'num_assignments': len(scenario.assignments),
            'assignments': [
                {
                    'task_id': assignment.task_id,
                    'resource_id': assignment.resource_id,
                    'start_time': assignment.start_time,
                    'end_time': assignment.end_time,
                    'hours_allocated': assignment.hours_allocated
                } for assignment in scenario.assignments
            ]
        }
        
        return ScenarioMetrics(
            scenario_id=scenario.id,
            total_time_days=total_time_days,
            total_cost=total_cost,
            quality_score=quality_score,
            resource_utilization=resource_utilization,
            parallelization_factor=parallelization_factor,
            skill_match_score=skill_match_score,
            constraint_violations=constraint_violations,
            overall_score=overall_score
        )
    
    def _calculate_resource_utilization(self, scenario: Scenario) -> float:
        """Calculate how efficiently resources are utilized"""
        if not scenario.assignments:
            return 0.0
        
        # Calculate resource usage
        resource_hours = {}
        for assignment in scenario.assignments:
            if assignment.resource_id not in resource_hours:
                resource_hours[assignment.resource_id] = 0
            resource_hours[assignment.resource_id] += assignment.hours_allocated
        
        # Calculate utilization rate
        total_available_hours = 0
        total_used_hours = 0
        
        for resource in self.project.resources:
            # Assuming project duration in days
            project_days = scenario.total_duration_hours / 8
            available_hours = resource.max_hours_per_day * project_days
            used_hours = resource_hours.get(resource.id, 0)
            
            total_available_hours += available_hours
            total_used_hours += used_hours
        
        return total_used_hours / total_available_hours if total_available_hours > 0 else 0
    
    def _calculate_parallelization(self, scenario: Scenario) -> float:
        """Calculate degree of task parallelization"""
        if not scenario.assignments:
            return 0.0
        
        # Find overlapping time periods
        time_slots = {}
        for assignment in scenario.assignments:
            for hour in range(int(assignment.start_time), int(assignment.end_time)):
                if hour not in time_slots:
                    time_slots[hour] = 0
                time_slots[hour] += 1
        
        # Calculate average parallelization
        if time_slots:
            avg_parallel_tasks = np.mean(list(time_slots.values()))
            max_parallel_tasks = max(time_slots.values())
            return avg_parallel_tasks / max_parallel_tasks if max_parallel_tasks > 0 else 0
        
        return 0.0
    
    def _calculate_skill_match(self, scenario: Scenario) -> float:
        """Calculate average skill match across all assignments"""
        if not scenario.assignments:
            return 0.0
        
        total_match = 0.0
        count = 0
        
        for assignment in scenario.assignments:
            task = next((t for t in self.project.tasks if t.id == assignment.task_id), None)
            resource = next((r for r in self.project.resources if r.id == assignment.resource_id), None)
            
            if task and resource:
                match_score = task.skill_match_score(resource)
                total_match += match_score
                count += 1
        
        return total_match / count if count > 0 else 0
    
    def _count_constraint_violations(self, scenario: Scenario) -> int:
        """Count number of constraint violations"""
        violations = 0
        
        # Budget constraint
        if self.project.constraints.max_budget:
            if scenario.total_cost > self.project.constraints.max_budget:
                violations += 1
        
        # Duration constraint
        if self.project.constraints.max_duration_days:
            max_hours = self.project.constraints.max_duration_days * 8
            if scenario.total_duration_hours > max_hours:
                violations += 1
        
        # Quality constraint
        if self.project.constraints.quality_gates:
            if scenario.quality_score < self.project.constraints.min_quality_score:
                violations += 1
        
        return violations
    
    def _calculate_overall_score(
        self,
        time_days: float,
        cost: float,
        quality: float,
        utilization: float,
        parallelization: float,
        violations: int
    ) -> float:
        """Calculate weighted overall score"""
        # Normalize metrics
        time_score = 1.0 / (1.0 + time_days / 30)  # Assuming 30 days as baseline
        cost_score = 1.0 / (1.0 + cost / self.project.metadata.estimated_budget)
        
        # Weighted combination
        score = (
            0.25 * time_score +
            0.25 * cost_score +
            0.20 * quality +
            0.15 * utilization +
            0.15 * parallelization
        )
        
        # Penalty for constraint violations
        score *= (0.8 ** violations)
        
        return score
    
    def find_pareto_frontier(self, scenarios: List[Scenario]) -> List[Scenario]:
        """
        Find Pareto optimal scenarios (non-dominated solutions)
        Optimizing for: minimize time, minimize cost, maximize quality
        """
        if not scenarios:
            return []
        
        # Evaluate all scenarios
        self.all_metrics = [self.evaluate_scenario(s) for s in scenarios]
        
        # Find Pareto frontier
        pareto_scenarios = []
        
        for i, scenario in enumerate(scenarios):
            metrics_i = self.all_metrics[i]
            is_dominated = False
            
            for j, other_scenario in enumerate(scenarios):
                if i == j:
                    continue
                
                metrics_j = self.all_metrics[j]
                
                # Check if scenario j dominates scenario i
                # (better in all objectives or equal in all and better in at least one)
                time_better = metrics_j.total_time_days <= metrics_i.total_time_days
                cost_better = metrics_j.total_cost <= metrics_i.total_cost
                quality_better = metrics_j.quality_score >= metrics_i.quality_score
                
                strictly_better = (
                    (metrics_j.total_time_days < metrics_i.total_time_days) or
                    (metrics_j.total_cost < metrics_i.total_cost) or
                    (metrics_j.quality_score > metrics_i.quality_score)
                )
                
                if time_better and cost_better and quality_better and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_scenarios.append(scenario)
        
        self.pareto_frontier = pareto_scenarios
        return pareto_scenarios
    
    def rank_scenarios(self, scenarios: List[Scenario]) -> List[Tuple[Scenario, ScenarioMetrics]]:
        """Rank scenarios by overall score"""
        scenario_metrics = []
        
        for scenario in scenarios:
            metrics = self.evaluate_scenario(scenario)
            scenario_metrics.append((scenario, metrics))
        
        # Sort by overall score (descending)
        scenario_metrics.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return scenario_metrics
    
    def get_best_scenario_by_objective(
        self,
        scenarios: List[Scenario],
        objective: str
    ) -> Optional[Tuple[Scenario, ScenarioMetrics]]:
        """Get best scenario for specific objective"""
        if not scenarios:
            return None
        
        metrics_list = [(s, self.evaluate_scenario(s)) for s in scenarios]
        
        if objective == 'time':
            return min(metrics_list, key=lambda x: x[1].total_time_days)
        elif objective == 'cost':
            return min(metrics_list, key=lambda x: x[1].total_cost)
        elif objective == 'quality':
            return max(metrics_list, key=lambda x: x[1].quality_score)
        elif objective == 'balanced':
            return max(metrics_list, key=lambda x: x[1].overall_score)
        
        return None
    
    def visualize_pareto_frontier(
        self,
        scenarios: List[Scenario],
        save_path: Optional[str] = None
    ):
        """Visualize Pareto frontier in 2D (time vs cost)"""
        if not scenarios:
            return
        
        # Get metrics
        metrics = [self.evaluate_scenario(s) for s in scenarios]
        
        # Find Pareto frontier
        pareto_scenarios = self.find_pareto_frontier(scenarios)
        pareto_metrics = [self.evaluate_scenario(s) for s in pareto_scenarios]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Time vs Cost
        all_times = [m.total_time_days for m in metrics]
        all_costs = [m.total_cost for m in metrics]
        pareto_times = [m.total_time_days for m in pareto_metrics]
        pareto_costs = [m.total_cost for m in pareto_metrics]
        
        ax1.scatter(all_times, all_costs, alpha=0.5, label='All Scenarios')
        ax1.scatter(pareto_times, pareto_costs, color='red', s=100, 
                   label='Pareto Optimal', zorder=5)
        
        # Sort Pareto points and draw frontier line
        if pareto_times:
            pareto_points = sorted(zip(pareto_times, pareto_costs))
            pareto_times_sorted = [p[0] for p in pareto_points]
            pareto_costs_sorted = [p[1] for p in pareto_points]
            ax1.plot(pareto_times_sorted, pareto_costs_sorted, 'r--', alpha=0.5)
        
        ax1.set_xlabel('Total Time (days)')
        ax1.set_ylabel('Total Cost ($)')
        ax1.set_title('Pareto Frontier: Time vs Cost')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quality Score Distribution
        quality_scores = [m.quality_score for m in metrics]
        scenario_names = [s.optimization_type for s in scenarios]
        
        bars = ax2.bar(range(len(quality_scores)), quality_scores)
        
        # Color Pareto optimal scenarios
        for i, scenario in enumerate(scenarios):
            if scenario in pareto_scenarios:
                bars[i].set_color('red')
        
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Quality Scores by Scenario')
        ax2.set_xticks(range(len(scenario_names)))
        ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, scenarios: List[Scenario]) -> Dict:
        """Generate comprehensive report of scenario analysis"""
        # Rank scenarios
        ranked = self.rank_scenarios(scenarios)
        
        # Find Pareto frontier
        pareto = self.find_pareto_frontier(scenarios)
        
        # Get best by objective
        best_time = self.get_best_scenario_by_objective(scenarios, 'time')
        best_cost = self.get_best_scenario_by_objective(scenarios, 'cost')
        best_quality = self.get_best_scenario_by_objective(scenarios, 'quality')
        best_balanced = self.get_best_scenario_by_objective(scenarios, 'balanced')
        
        report = {
            'summary': {
                'total_scenarios_evaluated': len(scenarios),
                'pareto_optimal_scenarios': len(pareto),
                'estimated_budget': self.project.metadata.estimated_budget
            },
            'best_scenarios': {
                'fastest': {
                    'scenario': best_time[0].to_dict() if best_time else None,
                    'metrics': best_time[1].to_dict() if best_time else None
                },
                'cheapest': {
                    'scenario': best_cost[0].to_dict() if best_cost else None,
                    'metrics': best_cost[1].to_dict() if best_cost else None
                },
                'highest_quality': {
                    'scenario': best_quality[0].to_dict() if best_quality else None,
                    'metrics': best_quality[1].to_dict() if best_quality else None
                },
                'best_balanced': {
                    'scenario': best_balanced[0].to_dict() if best_balanced else None,
                    'metrics': best_balanced[1].to_dict() if best_balanced else None
                }
            },
            'all_scenarios_ranked': [
                {
                    'rank': i + 1,
                    'scenario': scenario.to_dict(),
                    'metrics': metrics.to_dict()
                }
                for i, (scenario, metrics) in enumerate(ranked)
            ],
            'pareto_frontier': [
                {
                    'scenario': scenario.to_dict(),
                    'metrics': self.evaluate_scenario(scenario).to_dict()
                }
                for scenario in pareto
            ]
        }
        
        return report

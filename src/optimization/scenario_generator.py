"""
What-If Scenario Generator with various optimization strategies
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import random
from dataclasses import dataclass
import itertools

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.data_models import Project, Task, Resource, Scenario, TaskAssignment
from src.models.cms_transformer import validate_cms_data, get_cms_transformation_summary
from environment.scheduling_env import TaskSchedulingEnv
from agents.dqn_agent import DQNAgent


class ScenarioGenerator:
    """Generate and evaluate what-if scenarios for project optimization"""
    
    def __init__(self, project: Project):
        """
        Initialize scenario generator
        
        Args:
            project: Project data
        """
        self.project = project
        self.scenarios = []
        
    def generate_baseline_scenario(self) -> Scenario:
        """Generate baseline scenario with sequential task execution"""
        assignments = []
        current_time = 0.0
        total_cost = 0.0
        
        # Sort tasks by order
        sorted_tasks = sorted(self.project.tasks, key=lambda t: t.order)
        
        for task in sorted_tasks:
            # Find best resource for task (highest skill match)
            best_resource = None
            best_score = 0.0
            
            for resource in self.project.resources:
                if task.can_be_done_by(resource):
                    score = task.skill_match_score(resource)
                    if score > best_score:
                        best_score = score
                        best_resource = resource
            
            if best_resource:
                # Create assignment
                assignment = TaskAssignment(
                    task_id=task.id,
                    resource_id=best_resource.id,
                    start_time=current_time,
                    end_time=current_time + task.duration_hours,
                    hours_allocated=task.duration_hours
                )
                assignments.append(assignment)
                
                # Update time and cost
                current_time += task.duration_hours
                total_cost += task.duration_hours * best_resource.hourly_rate
        
        quality_score = self._calculate_quality_score(assignments)
        
        return Scenario(
            id="baseline_sequential",
            name="Baseline Sequential Execution",
            assignments=assignments,
            total_duration_hours=current_time,
            total_cost=total_cost,
            quality_score=quality_score,
            constraints_satisfied=self._check_constraints(total_cost, current_time),
            optimization_type="baseline"
        )
    
    def generate_parallel_scenario(self) -> Scenario:
        """Generate scenario with maximum parallelization"""
        assignments = []
        total_cost = 0.0
        
        # Track resource schedules
        resource_schedules = {res.id: [] for res in self.project.resources}
        task_start_times = {}
        task_end_times = {}
        
        # Group tasks by order for parallel execution
        order_groups = {}
        for task in self.project.tasks:
            if task.order not in order_groups:
                order_groups[task.order] = []
            order_groups[task.order].append(task)
        
        # Process each order group
        for order in sorted(order_groups.keys()):
            group_tasks = order_groups[order]
            
            # Get earliest possible start time for this group
            group_start_time = 0.0
            if order > 1:
                # Check dependencies from previous orders
                prev_order_tasks = [t for t in self.project.tasks if t.order < order]
                if prev_order_tasks:
                    completed_times = [task_end_times.get(t.id, 0) for t in prev_order_tasks]
                    group_start_time = max(completed_times) if completed_times else 0
            
            # For parallel execution, all tasks in the same order can start at the same time
            # if resources are available
            group_end_times = []
            
            # Assign tasks in parallel within the same order group
            for task in group_tasks:
                # Find available resource with best skill match
                best_resource = None
                best_start_time = group_start_time  # Start at group start time for parallel execution
                best_score = 0.0
                
                for resource in self.project.resources:
                    if task.can_be_done_by(resource):
                        score = task.skill_match_score(resource)
                        
                        # Find earliest available time for this resource after group start
                        resource_available_time = group_start_time
                        for scheduled in resource_schedules[resource.id]:
                            if scheduled['end'] > resource_available_time:
                                resource_available_time = scheduled['end']
                        
                        # For parallel execution, prefer resources available at group start time
                        if resource_available_time <= group_start_time:
                            resource_available_time = group_start_time
                        
                        # Consider this resource if it has better skill match or is available sooner
                        if score > best_score or (score == best_score and resource_available_time < best_start_time):
                            best_resource = resource
                            best_start_time = resource_available_time
                            best_score = score
                
                if best_resource:
                    # Create assignment
                    start_time = best_start_time
                    end_time = start_time + task.duration_hours
                    
                    assignment = TaskAssignment(
                        task_id=task.id,
                        resource_id=best_resource.id,
                        start_time=start_time,
                        end_time=end_time,
                        hours_allocated=task.duration_hours
                    )
                    assignments.append(assignment)
                    
                    # Update schedules
                    resource_schedules[best_resource.id].append({
                        'start': start_time,
                        'end': end_time,
                        'task_id': task.id
                    })
                    task_start_times[task.id] = start_time
                    task_end_times[task.id] = end_time
                    group_end_times.append(end_time)
                    
                    # Update cost
                    total_cost += task.duration_hours * best_resource.hourly_rate
        
        # Calculate total duration - maximum end time across all tasks
        total_duration = max(task_end_times.values()) if task_end_times else 0
        quality_score = self._calculate_quality_score(assignments)
        
        return Scenario(
            id="parallel_execution",
            name="Maximum Parallel Execution",
            assignments=assignments,
            total_duration_hours=total_duration,
            total_cost=total_cost,
            quality_score=quality_score,
            constraints_satisfied=self._check_constraints(total_cost, total_duration),
            optimization_type="time"
        )
    
    def generate_cost_optimized_scenario(self) -> Scenario:
        """Generate scenario optimized for minimum cost"""
        assignments = []
        total_cost = 0.0
        
        # Sort resources by hourly rate (cheapest first)
        sorted_resources = sorted(self.project.resources, key=lambda r: r.hourly_rate)
        
        # Sort tasks by order
        sorted_tasks = sorted(self.project.tasks, key=lambda t: t.order)
        
        # Track resource schedules
        resource_schedules = {res.id: 0.0 for res in self.project.resources}
        
        for task in sorted_tasks:
            # Find cheapest capable resource
            assigned = False
            
            for resource in sorted_resources:
                if task.can_be_done_by(resource):
                    # Get resource availability
                    start_time = resource_schedules[resource.id]
                    
                    # Check dependencies
                    for prev_task in sorted_tasks:
                        if prev_task.order < task.order:
                            prev_assignment = next(
                                (a for a in assignments if a.task_id == prev_task.id), None
                            )
                            if prev_assignment:
                                start_time = max(start_time, prev_assignment.end_time)
                    
                    # Create assignment
                    assignment = TaskAssignment(
                        task_id=task.id,
                        resource_id=resource.id,
                        start_time=start_time,
                        end_time=start_time + task.duration_hours,
                        hours_allocated=task.duration_hours
                    )
                    assignments.append(assignment)
                    
                    # Update schedule and cost
                    resource_schedules[resource.id] = start_time + task.duration_hours
                    total_cost += task.duration_hours * resource.hourly_rate
                    assigned = True
                    break
            
            if not assigned:
                # Fallback to any capable resource
                for resource in self.project.resources:
                    if task.can_be_done_by(resource):
                        start_time = resource_schedules[resource.id]
                        assignment = TaskAssignment(
                            task_id=task.id,
                            resource_id=resource.id,
                            start_time=start_time,
                            end_time=start_time + task.duration_hours,
                            hours_allocated=task.duration_hours
                        )
                        assignments.append(assignment)
                        resource_schedules[resource.id] = start_time + task.duration_hours
                        total_cost += task.duration_hours * resource.hourly_rate
                        break
        
        # Calculate total duration
        total_duration = max(a.end_time for a in assignments) if assignments else 0
        quality_score = self._calculate_quality_score(assignments)
        
        return Scenario(
            id="cost_optimized",
            name="Cost Optimized Execution",
            assignments=assignments,
            total_duration_hours=total_duration,
            total_cost=total_cost,
            quality_score=quality_score,
            constraints_satisfied=self._check_constraints(total_cost, total_duration),
            optimization_type="cost"
        )
    
    def generate_balanced_scenario(self) -> Scenario:
        """Generate balanced scenario optimizing both time and cost"""
        # Use weighted scoring to balance time and cost
        best_scenario = None
        best_score = float('-inf')
        
        # Generate multiple candidate scenarios
        for time_weight in [0.3, 0.5, 0.7]:
            cost_weight = 1 - time_weight
            
            assignments = []
            total_cost = 0.0
            
            # Track resource schedules
            resource_schedules = {res.id: [] for res in self.project.resources}
            
            # Sort tasks by order
            sorted_tasks = sorted(self.project.tasks, key=lambda t: t.order)
            
            for task in sorted_tasks:
                # Score each resource based on time and cost
                best_resource = None
                best_score_local = float('-inf')
                best_start_time = 0.0
                
                for resource in self.project.resources:
                    if task.can_be_done_by(resource):
                        # Calculate resource availability
                        resource_end_times = [s['end'] for s in resource_schedules[resource.id]]
                        resource_available = max(resource_end_times) if resource_end_times else 0
                        
                        # Check dependencies
                        dep_time = 0.0
                        for prev_task in sorted_tasks:
                            if prev_task.order < task.order:
                                prev_assignment = next(
                                    (a for a in assignments if a.task_id == prev_task.id), None
                                )
                                if prev_assignment:
                                    dep_time = max(dep_time, prev_assignment.end_time)
                        
                        start_time = max(resource_available, dep_time)
                        
                        # Score based on time and cost
                        time_score = 1.0 / (1.0 + start_time)  # Earlier is better
                        cost_score = 1.0 / (1.0 + resource.hourly_rate)  # Cheaper is better
                        skill_score = task.skill_match_score(resource)
                        
                        combined_score = (
                            time_weight * time_score +
                            cost_weight * cost_score +
                            0.2 * skill_score
                        )
                        
                        if combined_score > best_score_local:
                            best_resource = resource
                            best_score_local = combined_score
                            best_start_time = start_time
                
                if best_resource:
                    # Create assignment
                    assignment = TaskAssignment(
                        task_id=task.id,
                        resource_id=best_resource.id,
                        start_time=best_start_time,
                        end_time=best_start_time + task.duration_hours,
                        hours_allocated=task.duration_hours
                    )
                    assignments.append(assignment)
                    
                    # Update schedule
                    resource_schedules[best_resource.id].append({
                        'start': best_start_time,
                        'end': best_start_time + task.duration_hours,
                        'task_id': task.id
                    })
                    
                    # Update cost
                    total_cost += task.duration_hours * best_resource.hourly_rate
            
            # Calculate metrics
            total_duration = max(a.end_time for a in assignments) if assignments else 0
            quality_score = self._calculate_quality_score(assignments)
            
            # Score the scenario
            scenario_score = self._calculate_scenario_score(
                total_duration, total_cost, quality_score, time_weight, cost_weight
            )
            
            if scenario_score > best_score:
                best_score = scenario_score
                best_scenario = Scenario(
                    id=f"balanced_{time_weight}",
                    name=f"Balanced Optimization (Time: {time_weight:.1f}, Cost: {cost_weight:.1f})",
                    assignments=assignments,
                    total_duration_hours=total_duration,
                    total_cost=total_cost,
                    quality_score=quality_score,
                    constraints_satisfied=self._check_constraints(total_cost, total_duration),
                    optimization_type="balanced"
                )
        
        return best_scenario
    
    def generate_rl_optimized_scenario(
        self,
        optimization_mode: str = "balanced",
        num_episodes: int = 100
    ) -> Scenario:
        """Generate scenario using trained RL agent"""
        print(f"    Training RL agent for {optimization_mode} optimization...")
        env = TaskSchedulingEnv(self.project, optimization_mode)
        
        # Initialize DQN agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = DQNAgent(state_dim, action_dim)
        
        # Train agent with progress tracking
        best_reward = -float('inf')
        no_improvement_count = 0
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            step_count = 0
            max_steps = 100  # Prevent infinite loops
            
            while not done and step_count < max_steps:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train every few steps
                if len(agent.replay_buffer) >= agent.batch_size and step_count % 4 == 0:
                    agent.train_step()
                
                state = next_state
                total_reward += reward
                step_count += 1
            
            # Progress indicator every 10 episodes
            if episode % 10 == 0:
                print(f"      Episode {episode}/{num_episodes}, Reward: {total_reward:.2f}")
            
            # Early stopping with improvement tracking
            if total_reward > best_reward:
                best_reward = total_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Stop if converged or good performance
            if total_reward > 150 or no_improvement_count > 20:
                print(f"      Early stopping at episode {episode}")
                break
        
        if best_reward > -float('inf'):
            best_scenario = env.get_scenario()
            best_scenario.id = f"rl_{optimization_mode}"
            best_scenario.name = f"RL Optimized ({optimization_mode.capitalize()})"
        
        return best_scenario
    
    def _calculate_quality_score(self, assignments: List[TaskAssignment]) -> float:
        """Calculate quality score for assignments"""
        if not assignments:
            return 0.0
        
        total_score = 0.0
        for assignment in assignments:
            task = next((t for t in self.project.tasks if t.id == assignment.task_id), None)
            resource = next((r for r in self.project.resources if r.id == assignment.resource_id), None)
            
            if task and resource:
                skill_score = task.skill_match_score(resource)
                total_score += skill_score
        
        return total_score / len(assignments)
    
    def _check_constraints(self, cost: float, duration: float) -> bool:
        """Check if constraints are satisfied"""
        # Budget constraint
        if self.project.constraints.max_budget and cost > self.project.constraints.max_budget:
            return False
        
        # Duration constraint
        if self.project.constraints.max_duration_days:
            max_hours = self.project.constraints.max_duration_days * 8
            if duration > max_hours:
                return False
        
        return True
    
    def _calculate_scenario_score(
        self,
        duration: float,
        cost: float,
        quality: float,
        time_weight: float,
        cost_weight: float
    ) -> float:
        """Calculate overall scenario score"""
        # Normalize metrics
        time_score = 1.0 / (1.0 + duration / 100)
        cost_score = 1.0 / (1.0 + cost / 10000)
        
        # Combined score
        return (
            time_weight * time_score +
            cost_weight * cost_score +
            0.2 * quality
        )
    
    def generate_critical_path_scenario(self) -> Scenario:
        """Generate scenario using Critical Path Method (CPM) optimization"""
        assignments = []
        total_cost = 0.0
        
        # Build task dependency graph
        task_dependencies = {}
        for task in self.project.tasks:
            predecessors = [t for t in self.project.tasks if t.order < task.order]
            task_dependencies[task.id] = predecessors
        
        # Calculate earliest start and finish times
        earliest_start = {}
        earliest_finish = {}
        
        sorted_tasks = sorted(self.project.tasks, key=lambda t: t.order)
        
        for task in sorted_tasks:
            # Find earliest start based on predecessors
            if not task_dependencies[task.id]:
                earliest_start[task.id] = 0
            else:
                max_pred_finish = max(
                    earliest_finish.get(pred.id, 0) 
                    for pred in task_dependencies[task.id]
                )
                earliest_start[task.id] = max_pred_finish
            
            # Assign best resource for critical tasks
            best_resource = None
            best_score = -1
            
            for resource in self.project.resources:
                if task.can_be_done_by(resource):
                    # Prioritize skill match and speed
                    skill_score = task.skill_match_score(resource)
                    speed_score = 1.0 / (resource.hourly_rate / 100)  # Assume higher rate = faster
                    combined_score = skill_score * 0.7 + speed_score * 0.3
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_resource = resource
            
            if best_resource:
                assignment = TaskAssignment(
                    task_id=task.id,
                    resource_id=best_resource.id,
                    start_time=earliest_start[task.id],
                    end_time=earliest_start[task.id] + task.duration_hours,
                    hours_allocated=task.duration_hours
                )
                assignments.append(assignment)
                earliest_finish[task.id] = assignment.end_time
                total_cost += task.duration_hours * best_resource.hourly_rate
        
        total_duration = max(earliest_finish.values()) if earliest_finish else 0
        quality_score = self._calculate_quality_score(assignments)
        
        return Scenario(
            id="critical_path",
            name="Critical Path Optimized",
            assignments=assignments,
            total_duration_hours=total_duration,
            total_cost=total_cost,
            quality_score=quality_score,
            constraints_satisfied=self._check_constraints(total_cost, total_duration),
            optimization_type="critical_path"
        )
    
    def generate_resource_leveling_scenario(self) -> Scenario:
        """Generate scenario with resource leveling to avoid overallocation"""
        assignments = []
        total_cost = 0.0
        
        # Track resource availability per time unit
        resource_calendar = {res.id: [] for res in self.project.resources}
        sorted_tasks = sorted(self.project.tasks, key=lambda t: t.order)
        
        for task in sorted_tasks:
            best_assignment = None
            min_end_time = float('inf')
            
            # Try each capable resource
            for resource in self.project.resources:
                if task.can_be_done_by(resource):
                    # Find earliest available slot for this resource
                    if not resource_calendar[resource.id]:
                        start_time = 0
                    else:
                        start_time = max(resource_calendar[resource.id])
                    
                    # Check dependencies
                    for prev_task in self.project.tasks:
                        if prev_task.order < task.order:
                            prev_assignment = next(
                                (a for a in assignments if a.task_id == prev_task.id), None
                            )
                            if prev_assignment:
                                start_time = max(start_time, prev_assignment.end_time)
                    
                    end_time = start_time + task.duration_hours
                    
                    # Choose resource that can complete task earliest
                    if end_time < min_end_time:
                        min_end_time = end_time
                        best_assignment = TaskAssignment(
                            task_id=task.id,
                            resource_id=resource.id,
                            start_time=start_time,
                            end_time=end_time,
                            hours_allocated=task.duration_hours
                        )
                        best_resource = resource
            
            if best_assignment:
                assignments.append(best_assignment)
                resource_calendar[best_assignment.resource_id].append(best_assignment.end_time)
                total_cost += task.duration_hours * best_resource.hourly_rate
        
        total_duration = max(a.end_time for a in assignments) if assignments else 0
        quality_score = self._calculate_quality_score(assignments)
        
        return Scenario(
            id="resource_leveling",
            name="Resource Leveling Optimized",
            assignments=assignments,
            total_duration_hours=total_duration,
            total_cost=total_cost,
            quality_score=quality_score,
            constraints_satisfied=self._check_constraints(total_cost, total_duration),
            optimization_type="resource_leveling"
        )
    
    def generate_custom_parallel_scenario(self, task_constraints: Dict) -> Scenario:
        """Generate scenario with custom parallel execution constraints"""
        assignments = []
        total_cost = 0.0
        
        # Track resource schedules
        resource_schedules = {res.id: [] for res in self.project.resources}
        task_start_times = {}
        task_end_times = {}
        
        # Group tasks by order, considering parallel execution flags
        order_groups = {}
        for task in self.project.tasks:
            task_id = task.id
            allow_parallel = task_constraints.get(task_id, {}).get('allow_parallel', False)
            
            # If parallel execution is allowed, group with other parallel tasks of same order
            # Otherwise, create separate sequential groups
            group_key = task.order if allow_parallel else f"{task.order}_{task_id}"
            
            if group_key not in order_groups:
                order_groups[group_key] = []
            order_groups[group_key].append(task)
        
        # Process groups in order
        sorted_groups = sorted(order_groups.items(), key=lambda x: float(str(x[0]).split('_')[0]))
        
        for group_key, group_tasks in sorted_groups:
            # Get earliest possible start time for this group
            group_start_time = 0.0
            
            # Check dependencies from previous order groups
            current_order = int(str(group_key).split('_')[0])
            if current_order > 1:
                prev_tasks = [t for t in self.project.tasks if t.order < current_order]
                if prev_tasks:
                    completed_times = [task_end_times.get(t.id, 0) for t in prev_tasks]
                    group_start_time = max(completed_times) if completed_times else 0
            
            # Check if this is a parallel group (multiple tasks with same order and parallel flag)
            is_parallel_group = len(group_tasks) > 1 and any(
                task_constraints.get(t.id, {}).get('allow_parallel', False) for t in group_tasks
            )
            
            if is_parallel_group:
                # Parallel execution: assign tasks to different resources simultaneously
                for task in group_tasks:
                    best_resource = None
                    best_start_time = group_start_time
                    best_score = 0.0
                    
                    for resource in self.project.resources:
                        if task.can_be_done_by(resource):
                            score = task.skill_match_score(resource)
                            
                            # Find earliest available time for this resource
                            resource_available_time = group_start_time
                            for scheduled in resource_schedules[resource.id]:
                                if scheduled['end'] > resource_available_time:
                                    resource_available_time = scheduled['end']
                            
                            # Prefer resources available at group start time for true parallelism
                            if resource_available_time <= group_start_time and score > best_score:
                                best_resource = resource
                                best_start_time = group_start_time
                                best_score = score
                            elif best_resource is None and score > best_score:
                                best_resource = resource
                                best_start_time = resource_available_time
                                best_score = score
                    
                    if best_resource:
                        # Apply custom duration if specified
                        duration = task_constraints.get(task.id, {}).get('duration_hours', task.duration_hours)
                        
                        assignment = TaskAssignment(
                            task_id=task.id,
                            resource_id=best_resource.id,
                            start_time=best_start_time,
                            end_time=best_start_time + duration,
                            hours_allocated=duration
                        )
                        assignments.append(assignment)
                        
                        # Update schedules
                        resource_schedules[best_resource.id].append({
                            'start': best_start_time,
                            'end': best_start_time + duration,
                            'task_id': task.id
                        })
                        task_start_times[task.id] = best_start_time
                        task_end_times[task.id] = best_start_time + duration
                        
                        total_cost += duration * best_resource.hourly_rate
            else:
                # Sequential execution within the group
                current_time = group_start_time
                for task in group_tasks:
                    best_resource = None
                    best_score = 0.0
                    
                    for resource in self.project.resources:
                        if task.can_be_done_by(resource):
                            score = task.skill_match_score(resource)
                            if score > best_score:
                                best_resource = resource
                                best_score = score
                    
                    if best_resource:
                        # Apply custom duration if specified
                        duration = task_constraints.get(task.id, {}).get('duration_hours', task.duration_hours)
                        
                        assignment = TaskAssignment(
                            task_id=task.id,
                            resource_id=best_resource.id,
                            start_time=current_time,
                            end_time=current_time + duration,
                            hours_allocated=duration
                        )
                        assignments.append(assignment)
                        
                        task_start_times[task.id] = current_time
                        task_end_times[task.id] = current_time + duration
                        current_time += duration
                        
                        total_cost += duration * best_resource.hourly_rate
        
        # Calculate total duration
        total_duration = max(task_end_times.values()) if task_end_times else 0
        quality_score = self._calculate_quality_score(assignments)
        
        return Scenario(
            id="custom_parallel",
            name="Custom Parallel Execution",
            assignments=assignments,
            total_duration_hours=total_duration,
            total_cost=total_cost,
            quality_score=quality_score,
            constraints_satisfied=self._check_constraints(total_cost, total_duration),
            optimization_type="custom"
        )
    
    def generate_all_scenarios(self, include_rl: bool = True) -> List[Scenario]:
        """Generate all scenario types"""
        scenarios = []
        
        # Generate different scenario types
        scenarios.append(self.generate_baseline_scenario())
        scenarios.append(self.generate_parallel_scenario())
        scenarios.append(self.generate_cost_optimized_scenario())
        scenarios.append(self.generate_balanced_scenario())
        
        # Add new optimization strategies
        scenarios.append(self.generate_critical_path_scenario())
        scenarios.append(self.generate_resource_leveling_scenario())
        
        if include_rl:
            # Generate RL-optimized scenarios with improved training
            modes = ['time', 'cost', 'balanced']
            for mode in modes:
                rl_scenario = self.generate_rl_optimized_scenario(mode, num_episodes=50)  # Reduced episodes
                if rl_scenario:
                    scenarios.append(rl_scenario)
        
        self.scenarios = scenarios
        return scenarios
    
    def create_cms_baseline_scenario(self, cms_data: Dict) -> Scenario:
        """Create baseline scenario from CMS process structure"""
        assignments = []
        current_time = 0.0
        total_cost = 0.0
        
        # Process tasks in CMS order
        process_tasks = cms_data.get('process_task', cms_data.get('process_tasks', []))
        for process_task in sorted(process_tasks, key=lambda x: x['order']):
            task = process_task['task']
            duration_hours = task['task_capacity_minutes'] / 60.0
            
            # Use existing job assignments from CMS
            for job_task in task['jobTasks']:
                job = job_task['job']
                
                assignment = TaskAssignment(
                    task_id=f"task_{task['task_id']:03d}",
                    resource_id=f"resource_{job['job_id']:03d}",
                    start_time=current_time,
                    end_time=current_time + duration_hours,
                    hours_allocated=duration_hours
                )
                assignments.append(assignment)
                
                # Calculate cost using actual CMS rates
                total_cost += duration_hours * job['hourlyRate']
            
            current_time += duration_hours  # Sequential by default
        
        return Scenario(
            id="cms_baseline",
            name="CMS Process Baseline",
            assignments=assignments,
            total_duration_hours=current_time,
            total_cost=total_cost,
            quality_score=0.85,  # Default quality score
            constraints_satisfied=True,
            optimization_type="cms_baseline"
        )
    
    def generate_cms_optimization_scenarios(self, baseline: Scenario, cms_data: Dict) -> List[Scenario]:
        """Generate optimized scenarios preserving CMS structure"""
        scenarios = [baseline]  # CMS baseline is first scenario
        
        # Parallel execution optimization (respects job assignments)
        parallel_scenario = self.optimize_cms_parallel_tasks(baseline, cms_data)
        scenarios.append(parallel_scenario)
        
        # Resource efficiency optimization
        efficient_scenario = self.optimize_cms_resource_utilization(baseline, cms_data)
        scenarios.append(efficient_scenario)
        
        # Critical path optimization
        critical_path_scenario = self.optimize_cms_critical_path(baseline, cms_data)
        scenarios.append(critical_path_scenario)
        
        return scenarios
    
    def optimize_cms_parallel_tasks(self, baseline: Scenario, cms_data: Dict) -> Scenario:
        """Optimize for parallel execution while keeping job assignments"""
        assignments = []
        total_cost = 0.0
        max_end_time = 0.0
        
        # Group tasks that can run in parallel (no dependencies)
        task_groups = []
        current_group = []
        
        process_tasks = cms_data.get('process_task', cms_data.get('process_tasks', []))
        for process_task in sorted(process_tasks, key=lambda x: x['order']):
            task = process_task['task']
            # Simple grouping: tasks with same or adjacent order can be parallel
            if not current_group or process_task['order'] - current_group[-1]['order'] <= 1:
                current_group.append(process_task)
            else:
                task_groups.append(current_group)
                current_group = [process_task]
        
        if current_group:
            task_groups.append(current_group)
        
        # Process each group
        current_time = 0.0
        for group in task_groups:
            group_max_duration = 0.0
            
            for process_task in group:
                task = process_task['task']
                duration_hours = task['task_capacity_minutes'] / 60.0
                group_max_duration = max(group_max_duration, duration_hours)
                
                # Keep original job assignment from CMS
                for job_task in task['jobTasks']:
                    job = job_task['job']
                    
                    assignment = TaskAssignment(
                        task_id=f"task_{task['task_id']:03d}",
                        resource_id=f"resource_{job['job_id']:03d}",
                        start_time=current_time,
                        end_time=current_time + duration_hours,
                        hours_allocated=duration_hours
                    )
                    assignments.append(assignment)
                    total_cost += duration_hours * job['hourlyRate']
            
            current_time += group_max_duration
            max_end_time = current_time
        
        return Scenario(
            id="cms_parallel",
            name="CMS Parallel Execution",
            assignments=assignments,
            total_duration_hours=max_end_time,
            total_cost=total_cost,
            quality_score=0.85,
            constraints_satisfied=True,
            optimization_type="parallel"
        )
    
    def optimize_cms_resource_utilization(self, baseline: Scenario, cms_data: Dict) -> Scenario:
        """Optimize resource utilization within CMS constraints"""
        assignments = []
        total_cost = 0.0
        
        # Track resource availability
        resource_availability = {}
        
        process_tasks = cms_data.get('process_task', cms_data.get('process_tasks', []))
        for process_task in sorted(process_tasks, key=lambda x: x['order']):
            task = process_task['task']
            duration_hours = task['task_capacity_minutes'] / 60.0
            
            # Find earliest available time for this task's resources
            earliest_start = 0.0
            for job_task in task['jobTasks']:
                job = job_task['job']
                resource_id = f"resource_{job['job_id']:03d}"
                
                if resource_id in resource_availability:
                    earliest_start = max(earliest_start, resource_availability[resource_id])
            
            # Assign task at earliest available time
            for job_task in task['jobTasks']:
                job = job_task['job']
                resource_id = f"resource_{job['job_id']:03d}"
                
                assignment = TaskAssignment(
                    task_id=f"task_{task['task_id']:03d}",
                    resource_id=resource_id,
                    start_time=earliest_start,
                    end_time=earliest_start + duration_hours,
                    hours_allocated=duration_hours
                )
                assignments.append(assignment)
                
                # Update resource availability
                resource_availability[resource_id] = earliest_start + duration_hours
                total_cost += duration_hours * job['hourlyRate']
        
        # Calculate total duration
        total_duration = max(resource_availability.values()) if resource_availability else 0.0
        
        return Scenario(
            id="cms_resource_optimized",
            name="CMS Resource Optimized",
            assignments=assignments,
            total_duration_hours=total_duration,
            total_cost=total_cost,
            quality_score=0.88,
            constraints_satisfied=True,
            optimization_type="resource_optimized"
        )
    
    def optimize_cms_critical_path(self, baseline: Scenario, cms_data: Dict) -> Scenario:
        """Optimize critical path within CMS constraints"""
        assignments = []
        total_cost = 0.0
        
        # Identify critical tasks (longer duration, more resources)
        critical_tasks = []
        non_critical_tasks = []
        
        process_tasks = cms_data.get('process_task', cms_data.get('process_tasks', []))
        for process_task in sorted(process_tasks, key=lambda x: x['order']):
            task = process_task['task']
            duration_minutes = task['task_capacity_minutes']
            
            # Consider tasks > 30 minutes as critical
            if duration_minutes > 30:
                critical_tasks.append(process_task)
            else:
                non_critical_tasks.append(process_task)
        
        # Process critical tasks first
        current_time = 0.0
        for process_task in critical_tasks:
            task = process_task['task']
            duration_hours = task['task_capacity_minutes'] / 60.0
            
            for job_task in task['jobTasks']:
                job = job_task['job']
                
                assignment = TaskAssignment(
                    task_id=f"task_{task['task_id']:03d}",
                    resource_id=f"resource_{job['job_id']:03d}",
                    start_time=current_time,
                    end_time=current_time + duration_hours,
                    hours_allocated=duration_hours
                )
                assignments.append(assignment)
                total_cost += duration_hours * job['hourlyRate']
            
            current_time += duration_hours
        
        # Process non-critical tasks in parallel where possible
        parallel_start = current_time
        max_duration = 0.0
        
        for process_task in non_critical_tasks:
            task = process_task['task']
            duration_hours = task['task_capacity_minutes'] / 60.0
            max_duration = max(max_duration, duration_hours)
            
            for job_task in task['jobTasks']:
                job = job_task['job']
                
                assignment = TaskAssignment(
                    task_id=f"task_{task['task_id']:03d}",
                    resource_id=f"resource_{job['job_id']:03d}",
                    start_time=parallel_start,
                    end_time=parallel_start + duration_hours,
                    hours_allocated=duration_hours
                )
                assignments.append(assignment)
                total_cost += duration_hours * job['hourlyRate']
        
        total_duration = current_time + max_duration
        
        return Scenario(
            id="cms_critical_path",
            name="CMS Critical Path",
            assignments=assignments,
            total_duration_hours=total_duration,
            total_cost=total_cost,
            quality_score=0.90,
            constraints_satisfied=True,
            optimization_type="critical_path"
        )

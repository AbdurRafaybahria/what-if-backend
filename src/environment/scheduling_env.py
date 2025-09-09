"""
Reinforcement Learning Environment for Task Scheduling Optimization
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.data_models import Project, Task, Resource, TaskAssignment, Scenario


class TaskSchedulingEnv(gym.Env):
    """
    OpenAI Gym environment for task scheduling optimization
    
    State: Current assignments, resource availability, task completion status
    Action: Assign resource to task or skip
    Reward: Based on time, cost, and constraint satisfaction
    """
    
    def __init__(self, project: Project, optimization_mode: str = 'balanced'):
        """
        Initialize the environment
        
        Args:
            project: Project data
            optimization_mode: 'time', 'cost', or 'balanced'
        """
        super().__init__()
        self.project = project
        self.optimization_mode = optimization_mode
        
        # Define action and observation spaces
        self.n_tasks = len(project.tasks)
        self.n_resources = len(project.resources)
        
        # Action space: (task_id, resource_id, hours_to_allocate)
        # Simplified: discrete choice of task-resource pairs plus no-op
        self.action_space = spaces.Discrete(self.n_tasks * self.n_resources + 1)
        # Define enhanced observation space with constraint awareness
        # Tasks (completion + order) + Resources (availability + cost) + metrics + constraints + mode + progress
        obs_dim = (len(project.tasks) * 2) + (len(project.resources) * 2) + 2 + 2 + 3 + 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_time = 0.0
        self.current_cost = 0.0
        self.assignments = []
        self.task_completion = {task.id: 0.0 for task in self.project.tasks}
        self.resource_availability = {res.id: res.max_hours_per_day for res in self.project.resources}
        self.resource_daily_hours = {res.id: 0.0 for res in self.project.resources}
        self.completed_tasks = set()
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get enhanced state observation with constraint awareness"""
        obs = []
        
        # Task completion status
        for task in sorted(self.project.tasks, key=lambda t: t.id):
            obs.append(self.task_completion[task.id])
            # Task order as priority proxy
            obs.append(1.0 / (task.order + 1))  # Earlier tasks have higher priority
        
        # Resource availability and efficiency
        for resource in sorted(self.project.resources, key=lambda r: r.id):
            obs.append(self.resource_availability[resource.id] / 8.0)  # Normalize
            obs.append(resource.hourly_rate / 200.0)  # Normalize cost
        
        # Time and cost progress with constraint ratios
        obs.append(self.current_time / 100.0)  # Normalize
        obs.append(self.current_cost / 10000.0)  # Normalize
        
        # Constraint proximity indicators
        if self.project.constraints.max_budget:
            budget_usage = self.current_cost / self.project.constraints.max_budget
            obs.append(min(1.0, budget_usage))  # Budget usage ratio
        else:
            obs.append(0.0)
        
        if self.project.constraints.max_duration_days:
            time_usage = self.current_time / (self.project.constraints.max_duration_days * 8)
            obs.append(min(1.0, time_usage))  # Time usage ratio
        else:
            obs.append(0.0)
        
        # Add optimization mode encoding
        mode_encoding = {'time': [1, 0, 0], 'cost': [0, 1, 0], 'balanced': [0, 0, 1]}
        obs.extend(mode_encoding.get(self.optimization_mode, [0, 0, 1]))
        
        # Progress indicator
        obs.append(len(self.completed_tasks) / self.n_tasks)
        
        return np.array(obs, dtype=np.float32)
    
    def _decode_action(self, action: int) -> Optional[Tuple[str, str]]:
        """Decode discrete action to (task_id, resource_id) or None for no-op"""
        if action == self.n_tasks * self.n_resources:
            return None  # No-op action
        
        task_idx = action // self.n_resources
        resource_idx = action % self.n_resources
        
        if task_idx < len(self.project.tasks) and resource_idx < len(self.project.resources):
            return (self.project.tasks[task_idx].id, self.project.resources[resource_idx].id)
        return None
    
    def _calculate_reward(self, action_taken: bool) -> float:
        """Calculate enhanced reward based on current state and action"""
        reward = 0
        
        # Smaller time penalty to encourage exploration
        reward -= 0.05
        
        # Action bonus/penalty
        if action_taken:
            reward += 2.0  # Reward for taking valid actions
        else:
            reward -= 0.5  # Small penalty for invalid actions
        
        # Progressive task completion reward
        total_completion = sum(self.task_completion.values()) / self.n_tasks
        reward += total_completion * 15  # Increased from 10
        
        # Milestone rewards
        completed_ratio = len(self.completed_tasks) / self.n_tasks
        if 0.25 <= completed_ratio < 0.5:
            reward += 5
        elif 0.5 <= completed_ratio < 0.75:
            reward += 10
        elif 0.75 <= completed_ratio < 1.0:
            reward += 15
        
        # Optimization-specific rewards with better scaling
        if self.optimization_mode == 'time':
            # Strong reward for fast completion
            if self.current_time < 50:
                reward += 10
            time_efficiency = max(0, 100 - self.current_time) / 100
            reward += time_efficiency * 10
            
            # Parallelization bonus
            active_resources = sum(1 for h in self.resource_daily_hours.values() if h > 0)
            if active_resources > 1:
                reward += active_resources * 2
        
        elif self.optimization_mode == 'cost':
            # Strong reward for cost efficiency
            budget = self.project.metadata.estimated_budget
            if self.current_cost < budget * 0.5:
                reward += 15
            elif self.current_cost < budget * 0.8:
                reward += 8
            cost_efficiency = max(0, budget - self.current_cost) / budget
            reward += cost_efficiency * 10
        
        elif self.optimization_mode == 'balanced':
            # Balanced optimization with adaptive weighting
            budget = self.project.metadata.estimated_budget
            
            # Time component (40%)
            time_score = max(0, 100 - self.current_time) / 100
            reward += time_score * 8
            
            # Cost component (40%)
            cost_score = max(0, budget - self.current_cost) / budget
            reward += cost_score * 8
            
            # Quality component (20%)
            quality_score = total_completion
            reward += quality_score * 4
        
        # Super bonus for completing all tasks efficiently
        if len(self.completed_tasks) == self.n_tasks:
            reward += 150  # Increased from 100
            
            # Additional efficiency bonuses
            if self.current_time < 50:
                reward += 50
            if self.current_cost < self.project.metadata.estimated_budget * 0.5:
                reward += 50
        
        # Stronger penalties for constraint violations
        if self.project.constraints.max_budget and self.current_cost > self.project.constraints.max_budget:
            reward -= 30  # Increased from 20
        
        if self.project.constraints.max_duration_days and self.current_time > (self.project.constraints.max_duration_days * 8):
            reward -= 30  # Increased from 20    
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment
        
        Returns:
            observation, reward, done, info
        """
        decoded = self._decode_action(action)
        action_taken = False
        
        if decoded:
            task_id, resource_id = decoded
            
            # Check if action is valid
            task = next((t for t in self.project.tasks if t.id == task_id), None)
            resource = next((r for r in self.project.resources if r.id == resource_id), None)
            
            if task and resource and task_id not in self.completed_tasks:
                # Check if resource can do the task
                if task.can_be_done_by(resource):
                    # Check if there are predecessor dependencies
                    can_start = self._check_dependencies(task)
                    
                    if can_start and self.resource_availability[resource_id] > 0:
                        # Allocate resource to task
                        hours_to_allocate = min(
                            self.resource_availability[resource_id],
                            task.duration_hours * (1 - self.task_completion[task_id])
                        )
                        
                        if hours_to_allocate > 0:
                            # Create assignment
                            assignment = TaskAssignment(
                                task_id=task_id,
                                resource_id=resource_id,
                                start_time=self.current_time,
                                end_time=self.current_time + hours_to_allocate,
                                hours_allocated=hours_to_allocate
                            )
                            self.assignments.append(assignment)
                            
                            # Update state
                            self.task_completion[task_id] += hours_to_allocate / task.duration_hours
                            self.resource_availability[resource_id] -= hours_to_allocate
                            self.resource_daily_hours[resource_id] += hours_to_allocate
                            self.current_cost += hours_to_allocate * resource.hourly_rate
                            
                            # Mark task as completed if done
                            if self.task_completion[task_id] >= 0.999:
                                self.completed_tasks.add(task_id)
                                self.task_completion[task_id] = 1.0
                            
                            action_taken = True
        
        # Advance time if all resources are utilized or no valid actions
        if not action_taken or all(avail <= 0.1 for avail in self.resource_availability.values()):
            self._advance_time()
        
        # Calculate reward with constraint awareness
        base_reward = self._calculate_reward(action_taken)
        constraint_penalty = self._check_constraints_violation()
        reward = base_reward - constraint_penalty
        
        # Check if episode is done
        done = len(self.completed_tasks) == self.n_tasks or self.current_time > 1000
        
        # Prepare info
        info = {
            'completed_tasks': len(self.completed_tasks),
            'current_time': self.current_time,
            'current_cost': self.current_cost,
            'action_taken': action_taken
        }
        
        return self._get_observation(), reward, done, info
    
    def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        # Check if all tasks with lower order are completed
        for other_task in self.project.tasks:
            if other_task.order < task.order:
                if other_task.id not in self.completed_tasks:
                    return False
        return True
    
    def _check_constraints_violation(self) -> float:
        """Calculate penalty for constraint violations"""
        penalty = 0.0
        
        # Budget constraint check
        if self.project.constraints.max_budget:
            budget_ratio = self.current_cost / self.project.constraints.max_budget
            if budget_ratio > 0.9:  # Warning zone
                penalty += (budget_ratio - 0.9) * 10
            if budget_ratio > 1.0:  # Violation
                penalty += (budget_ratio - 1.0) * 50
        
        # Duration constraint check
        if self.project.constraints.max_duration_days:
            max_hours = self.project.constraints.max_duration_days * 8
            duration_ratio = self.current_time / max_hours
            if duration_ratio > 0.9:  # Warning zone
                penalty += (duration_ratio - 0.9) * 10
            if duration_ratio > 1.0:  # Violation
                penalty += (duration_ratio - 1.0) * 50
        
        # Resource overallocation check
        for resource_id, hours in self.resource_daily_hours.items():
            if hours > 8:  # Overtime penalty
                penalty += (hours - 8) * 2
        
        return penalty
    
    def _advance_time(self):
        """Advance to next time period (day)"""
        self.current_time += 8  # Advance by one working day
        # Reset daily resource availability
        for resource in self.project.resources:
            self.resource_availability[resource.id] = resource.max_hours_per_day
            self.resource_daily_hours[resource.id] = 0.0
    
    def get_scenario(self) -> Scenario:
        """Convert current state to a Scenario object"""
        quality_score = self._calculate_quality_score()
        constraints_satisfied = self._check_constraints()
        
        return Scenario(
            id=f"scenario_{self.optimization_mode}_{random.randint(1000, 9999)}",
            name=f"{self.optimization_mode.capitalize()} Optimization",
            assignments=self.assignments,
            total_duration_hours=self.current_time,
            total_cost=self.current_cost,
            quality_score=quality_score,
            constraints_satisfied=constraints_satisfied,
            optimization_type=self.optimization_mode
        )
    
    def _calculate_quality_score(self) -> float:
        """Calculate quality score based on skill matching and completion"""
        if not self.assignments:
            return 0.0
        
        total_score = 0.0
        for assignment in self.assignments:
            task = next((t for t in self.project.tasks if t.id == assignment.task_id), None)
            resource = next((r for r in self.project.resources if r.id == assignment.resource_id), None)
            
            if task and resource:
                skill_score = task.skill_match_score(resource)
                total_score += skill_score
        
        # Average skill match score
        avg_skill_score = total_score / len(self.assignments) if self.assignments else 0
        
        # Completion score
        completion_score = len(self.completed_tasks) / self.n_tasks
        
        # Combined quality score
        return 0.6 * avg_skill_score + 0.4 * completion_score
    
    def _check_constraints(self) -> bool:
        """Check if all constraints are satisfied"""
        # Budget constraint
        if self.project.constraints.max_budget and self.current_cost > self.project.constraints.max_budget:
            return False
        
        # Duration constraint
        if self.project.constraints.max_duration_days:
            max_hours = self.project.constraints.max_duration_days * 8
            if self.current_time > max_hours:
                return False
        
        # Quality gate
        if self.project.constraints.quality_gates:
            quality_score = self._calculate_quality_score()
            if quality_score < self.project.constraints.min_quality_score:
                return False
        
        return True

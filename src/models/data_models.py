"""
Data models for parsing and representing project data
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
from datetime import datetime


@dataclass
class Skill:
    """Represents a skill requirement or capability"""
    name: str
    level: int
    
    def matches(self, required_skill: 'Skill') -> bool:
        """Check if this skill meets the requirement"""
        return self.name == required_skill.name and self.level >= required_skill.level


@dataclass
class Task:
    """Represents a project task"""
    id: str
    name: str
    description: str
    duration_hours: float
    required_skills: List[Skill]
    order: int
    dependencies: List[str] = field(default_factory=list)
    
    def can_be_done_by(self, resource: 'Resource') -> bool:
        """Check if a resource has all required skills for this task"""
        for req_skill in self.required_skills:
            if not any(res_skill.matches(req_skill) for res_skill in resource.skills):
                return False
        return True
    
    def skill_match_score(self, resource: 'Resource') -> float:
        """Calculate how well a resource's skills match the task requirements"""
        if not self.can_be_done_by(resource):
            return 0.0
        
        total_score = 0.0
        for req_skill in self.required_skills:
            matching_skills = [s for s in resource.skills if s.name == req_skill.name]
            if matching_skills:
                # Higher score for overqualification
                skill_diff = matching_skills[0].level - req_skill.level
                total_score += 1.0 + (skill_diff * 0.2)
        
        return total_score / len(self.required_skills)


@dataclass
class Resource:
    """Represents a project resource (person)"""
    id: str
    name: str
    description: str
    skills: List[Skill]
    hourly_rate: float
    max_hours_per_day: float
    
    def daily_cost(self) -> float:
        """Calculate maximum daily cost for this resource"""
        return self.hourly_rate * self.max_hours_per_day


@dataclass
class ProjectConstraints:
    """Project constraints and requirements"""
    quality_gates: bool = True
    max_budget: Optional[float] = None
    max_duration_days: Optional[float] = None
    min_quality_score: float = 0.8


@dataclass
class ProjectMetadata:
    """Project metadata"""
    project_type: str
    complexity: str
    team_size: int
    estimated_budget: float


@dataclass
class Project:
    """Complete project representation"""
    id: str
    name: str
    description: str
    tasks: List[Task]
    resources: List[Resource]
    constraints: ProjectConstraints
    metadata: ProjectMetadata
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'Project':
        """Parse project from JSON data"""
        # Parse tasks
        tasks = []
        for task_data in json_data['tasks']:
            skills = [Skill(s['name'], s['level']) for s in task_data['required_skills']]
            task = Task(
                id=task_data['id'],
                name=task_data['name'],
                description=task_data['description'],
                duration_hours=task_data['duration_hours'],
                required_skills=skills,
                order=task_data['order'],
                dependencies=task_data.get('dependencies', [])
            )
            tasks.append(task)
        
        # Parse resources
        resources = []
        for res_data in json_data['resources']:
            skills = [Skill(s['name'], s['level']) for s in res_data['skills']]
            resource = Resource(
                id=res_data['id'],
                name=res_data['name'],
                description=res_data['description'],
                skills=skills,
                hourly_rate=res_data['hourly_rate'],
                max_hours_per_day=res_data['max_hours_per_day']
            )
            resources.append(resource)
        
        # Parse constraints
        constraints_data = json_data.get('constraints', {})
        constraints = ProjectConstraints(
            quality_gates=constraints_data.get('quality_gates', True),
            max_budget=constraints_data.get('max_budget'),
            max_duration_days=constraints_data.get('max_duration_days')
        )
        
        # Parse metadata
        metadata_data = json_data['metadata']
        metadata = ProjectMetadata(
            project_type=metadata_data['project_type'],
            complexity=metadata_data['complexity'],
            team_size=metadata_data['team_size'],
            estimated_budget=metadata_data['estimated_budget']
        )
        
        return cls(
            id=json_data['id'],
            name=json_data['name'],
            description=json_data['description'],
            tasks=tasks,
            resources=resources,
            constraints=constraints,
            metadata=metadata
        )
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'Project':
        """Load project from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_json(data)


@dataclass
class TaskAssignment:
    """Assignment of a resource to a task"""
    task_id: str
    resource_id: str
    start_time: float  # in hours from project start
    end_time: float
    hours_allocated: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def cost(self) -> float:
        """Cost needs to be calculated with resource hourly rate"""
        return 0.0  # Will be calculated in scenario


@dataclass
class Scenario:
    """A complete project execution scenario"""
    id: str
    name: str
    assignments: List[TaskAssignment]
    total_duration_hours: float
    total_cost: float
    quality_score: float
    constraints_satisfied: bool
    optimization_type: str  # 'time', 'cost', or 'balanced'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary for output"""
        return {
            'id': self.id,
            'name': self.name,
            'total_duration_days': self.total_duration_hours / 8,  # Convert to days
            'total_cost': self.total_cost,
            'quality_score': self.quality_score,
            'constraints_satisfied': self.constraints_satisfied,
            'optimization_type': self.optimization_type,
            'num_assignments': len(self.assignments),
            'assignments': [
                {
                    'task_id': assignment.task_id,
                    'resource_id': assignment.resource_id,
                    'start_time': assignment.start_time,
                    'end_time': assignment.end_time,
                    'hours_allocated': assignment.hours_allocated
                } for assignment in self.assignments
            ]
        }

"""
CMS Data Transformer for converting CMS process data to internal project format
"""
from typing import Dict, List, Any


def transform_cms_to_internal(cms_data: Dict) -> Dict:
    """
    Transform CMS process data to internal project format
    
    Args:
        cms_data: CMS process data with process_task structure
        
    Returns:
        Dict: Internal project format compatible with existing optimization system
    """
    
    # Extract tasks from process_task (handle both singular and plural)
    tasks = []
    resources = []
    resource_ids_seen = set()
    
    process_tasks = cms_data.get('process_task', cms_data.get('process_tasks', []))
    
    for process_task in process_tasks:
        task_data = process_task['task']
        
        # Convert minutes to hours
        duration_hours = task_data['task_capacity_minutes'] / 60.0
        
        # Create task in internal format
        task = {
            "id": f"task_{task_data['task_id']:03d}",
            "name": task_data['task_name'],
            "description": task_data['task_overview'],
            "duration_hours": duration_hours,
            "required_skills": [{"name": "general", "level": 3}],  # Default skill since CMS doesn't provide
            "order": process_task['order'],
            "dependencies": []  # CMS doesn't provide dependencies
        }
        tasks.append(task)
        
        # Extract resources from jobTasks (avoid duplicates)
        for job_task in task_data['jobTasks']:
            job = job_task['job']
            resource_id = f"resource_{job['job_id']:03d}"
            
            if resource_id not in resource_ids_seen:
                resource = {
                    "id": resource_id,
                    "name": job['name'],
                    "description": job['description'],
                    "skills": [{"name": "general", "level": 3}],  # Default skill
                    "hourly_rate": job['hourlyRate'],
                    "max_hours_per_day": job['maxHoursPerDay']
                }
                resources.append(resource)
                resource_ids_seen.add(resource_id)
    
    # Calculate estimated budget
    estimated_budget = sum(
        resource['hourly_rate'] * resource['max_hours_per_day'] * 30 
        for resource in resources
    )
    
    # Build internal project format
    internal_project = {
        "id": f"project_{cms_data['process_id']:03d}",
        "name": cms_data['process_name'],
        "description": cms_data['process_overview'],
        "tasks": tasks,
        "resources": resources,
        "constraints": {
            "quality_gates": True,
            "max_budget": None,
            "max_duration_days": None,
            "min_quality_score": 0.8
        },
        "metadata": {
            "project_type": "cms_import",
            "complexity": "medium",
            "team_size": len(resources),
            "estimated_budget": estimated_budget,
            "company_id": cms_data.get('company_id'),
            "process_code": cms_data.get('process_code'),
            "created_at": cms_data.get('created_at'),
            "updated_at": cms_data.get('updated_at')
        }
    }
    
    return internal_project


def validate_cms_data(cms_data: Dict) -> bool:
    """
    Validate that CMS data has required fields
    
    Args:
        cms_data: CMS process data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = [
        'process_id', 'process_name', 'process_overview'
    ]
    
    # Check top-level fields
    for field in required_fields:
        if field not in cms_data:
            return False
    
    # Check for either process_task or process_tasks
    process_tasks = cms_data.get('process_task', cms_data.get('process_tasks', []))
    if not isinstance(process_tasks, list) or len(process_tasks) == 0:
        return False
    
    for process_task in process_tasks:
        if 'task' not in process_task or 'order' not in process_task:
            return False
            
        task = process_task['task']
        required_task_fields = [
            'task_id', 'task_name', 'task_capacity_minutes', 'jobTasks'
        ]
        
        for field in required_task_fields:
            if field not in task:
                return False
        
        # Check jobTasks structure
        if not isinstance(task['jobTasks'], list):
            return False
            
        for job_task in task['jobTasks']:
            if 'job' not in job_task:
                return False
                
            job = job_task['job']
            required_job_fields = [
                'job_id', 'name', 'hourlyRate', 'maxHoursPerDay'
            ]
            
            for field in required_job_fields:
                if field not in job:
                    return False
    
    return True


def get_cms_transformation_summary(cms_data: Dict) -> Dict:
    """
    Get summary of CMS data transformation
    
    Args:
        cms_data: Original CMS data
        
    Returns:
        Dict: Summary of transformation details
    """
    process_tasks = cms_data.get('process_task', cms_data.get('process_tasks', []))
    total_tasks = len(process_tasks)
    total_duration_minutes = sum(
        task['task']['task_capacity_minutes'] 
        for task in process_tasks
    )
    
    unique_jobs = set()
    for process_task in process_tasks:
        for job_task in process_task['task']['jobTasks']:
            unique_jobs.add(job_task['job']['job_id'])
    
    return {
        "process_id": cms_data['process_id'],
        "process_name": cms_data['process_name'],
        "total_tasks": total_tasks,
        "total_duration_hours": total_duration_minutes / 60.0,
        "total_duration_days": (total_duration_minutes / 60.0) / 8,
        "unique_resources": len(unique_jobs),
        "company": cms_data.get('company', {}).get('name', 'Unknown')
    }

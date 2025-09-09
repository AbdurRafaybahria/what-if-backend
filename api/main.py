from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
from pathlib import Path
import sys
import os
import httpx
import asyncio

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.data_models import Project
from src.models.cms_transformer import transform_cms_to_internal, validate_cms_data, get_cms_transformation_summary
from src.optimization.scenario_generator import ScenarioGenerator
from src.optimization.pareto_optimizer import ParetoOptimizer

app = FastAPI(title="What-If Analysis API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomConstraints(BaseModel):
    resources: Dict[str, Dict[str, Any]]
    tasks: Dict[str, Dict[str, Any]]
    preferences: Dict[str, float]

class OptimizeRequest(BaseModel):
    process_name: str
    constraints: Optional[CustomConstraints] = None

# Available processes mapping
PROCESS_FILES = {
    "software_project": "example/software_project.json",
    "hospital_project": "example/hospital_project.json",
    "manufacturing_project": "example/manufacturing_project.json",
    "cms_ecommerce_project": "example/cms_ecommerce_project.json",
    "7": "example/software_project.json"
}

# Explicitly exclude cms-process from being handled by generic endpoint
EXCLUDED_PROCESS_NAMES = {"cms-process"}


@app.get("/")
async def root():
    return {"message": "What-If Analysis API is running"}

@app.get("/processes")
async def get_available_processes():
    """Get list of available processes including CMS processes"""
    processes = []
    
    # Try to fetch CMS processes first
    try:
        cms_processes = await get_cms_processes()
        if cms_processes:
            for process in cms_processes:
                processes.append({
                    "id": process['process_id'],
                    "name": process['process_name'],
                    "description": process.get('process_overview', ''),
                    "company": process.get('company', {}).get('name', 'Unknown'),
                    "tasks_count": len(process.get('process_tasks', [])),
                    "type": "cms",
                    "process_data": process
                })
    except Exception as e:
        print(f"Failed to load CMS processes: {e}")
    
    # Add local processes as fallback
    base_path = Path(__file__).parent.parent
    for process_id, file_path in PROCESS_FILES.items():
        full_path = base_path / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    data = json.load(f)
                processes.append({
                    "id": process_id,
                    "name": data.get("name", process_id),
                    "description": data.get("description", ""),
                    "company": "Local",
                    "tasks_count": len(data.get("tasks", [])),
                    "resources_count": len(data.get("resources", [])),
                    "type": "local"
                })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return {"processes": processes}





    raise HTTPException(status_code=404, detail="This endpoint has been removed. Use /optimize/cms-process/{process_id} instead")

@app.post("/optimize/cms-process/{process_id}")
async def optimize_cms_process_by_id(process_id: int):
    """Fetch CMS process by ID and optimize it"""
    try:
        # Fetch process data from CMS API
        cms_data = await get_cms_process_by_id(process_id)
        if not cms_data:
            raise HTTPException(status_code=404, detail=f"Process with ID {process_id} not found")
        
        # Validate CMS data structure
        if not validate_cms_data(cms_data):
            raise HTTPException(status_code=400, detail="Invalid CMS data format")
        
        # Transform CMS to internal format
        internal_data = transform_cms_to_internal(cms_data)
        
        # Create project from internal data
        project = Project.from_json(internal_data)
        
        # Initialize scenario generator
        generator = ScenarioGenerator(project)
        
        # Create CMS baseline scenario (preserves existing assignments)
        cms_baseline = generator.create_cms_baseline_scenario(cms_data)
        
        # Generate optimized scenarios from CMS baseline
        scenarios = generator.generate_cms_optimization_scenarios(cms_baseline, cms_data)
        
        return {
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "baseline": cms_baseline.to_dict(),
            "process_info": {
                "process_id": cms_data['process_id'],
                "process_name": cms_data['process_name'],
                "company": cms_data.get('company', {}).get('name', 'Unknown')
            }
        }
    except Exception as e:
        print(f"Error optimizing CMS process {process_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/optimize/custom")
async def optimize_with_custom_constraints(request: dict):
    """Optimize with custom constraints - accepts direct constraint payload"""
    print(f"DEBUG: Received custom optimization request")
    print(f"DEBUG: Request keys: {list(request.keys())}")
    
    # Extract constraints directly from request
    resources = request.get('resources', {})
    tasks = request.get('tasks', {})
    preferences = request.get('preferences', {})
    
    # Use process "7" as default for custom optimization
    process_name = "7"
    
    if process_name not in PROCESS_FILES:
        raise HTTPException(status_code=404, detail=f"Process not found: {process_name}")
    
    try:
        # Load base project
        base_path = Path(__file__).parent.parent
        file_path = base_path / PROCESS_FILES[process_name]
        
        with open(file_path, 'r') as f:
            project_data = json.load(f)
        
        # Apply custom constraints directly from request
        if resources or tasks:
            # Create a constraints object from the direct request data
            constraint_data = {
                'resources': resources,
                'tasks': tasks,
                'preferences': preferences
            }
            project_data = apply_direct_constraints(project_data, constraint_data)
        
        # Create temporary project with constraints
        project = Project.from_json(project_data)
        
        # Generate optimized scenario based on preferences and constraints
        generator = ScenarioGenerator(project)
        
        # Check if any tasks have parallel execution enabled
        has_parallel_tasks = False
        task_constraints = {}
        if tasks:
            for task_id, constraint in tasks.items():
                task_constraints[task_id] = constraint
                if constraint.get('allow_parallel', False):
                    has_parallel_tasks = True
        
        # Choose optimization type based on preferences
        time_pref = preferences.get('time_priority', 0.33)
        cost_pref = preferences.get('cost_priority', 0.33)
        quality_pref = preferences.get('quality_priority', 0.34)
        
        # Use custom parallel scenario if parallel tasks are configured
        if has_parallel_tasks:
            scenario = generator.generate_custom_parallel_scenario(task_constraints)
        elif time_pref > cost_pref and time_pref > quality_pref:
            scenario = generator.generate_parallel_scenario()
        elif cost_pref > time_pref and cost_pref > quality_pref:
            scenario = generator.generate_cost_optimized_scenario()
        else:
            scenario = generator.generate_balanced_scenario()
        
        # Evaluate scenario
        optimizer = ParetoOptimizer(project)
        metrics = optimizer.evaluate_scenario(scenario)
        
        result = {
            'scenario': scenario.to_dict(),
            'metrics': metrics.to_dict()
        }
        
        return {
            "success": True,
            "scenario": result,
            "project_data": project_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom optimization failed: {str(e)}")

@app.post("/optimize/{process_name}")
async def optimize_process(process_name: str):
    """Optimize a process and return the best scenario"""
    # Block cms-process completely
    if process_name == "cms-process":
        raise HTTPException(status_code=404, detail="Endpoint not found. Use /optimize/cms-process/{process_id} instead")
    
    # Block other excluded process names
    if process_name in EXCLUDED_PROCESS_NAMES:
        raise HTTPException(status_code=404, detail="Process not found")
        
    if process_name not in PROCESS_FILES:
        raise HTTPException(status_code=404, detail="Process not found")
    
    try:
        # Load project data
        base_path = Path(__file__).parent.parent
        file_path = base_path / PROCESS_FILES[process_name]
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Process file not found")
        
        # Load and parse project
        project = Project.from_json_file(str(file_path))
        
        # Generate scenarios
        generator = ScenarioGenerator(project)
        scenarios = [
            generator.generate_baseline_scenario(),
            generator.generate_parallel_scenario(),
            generator.generate_cost_optimized_scenario(),
            generator.generate_balanced_scenario(),
            generator.generate_critical_path_scenario(),
            generator.generate_resource_leveling_scenario()
        ]
        
        # Find best scenario using Pareto optimization
        optimizer = ParetoOptimizer(project)
        pareto_scenarios = []
        
        for scenario in scenarios:
            metrics = optimizer.evaluate_scenario(scenario)
            pareto_scenarios.append({
                'scenario': scenario.to_dict(),
                'metrics': metrics.to_dict()
            })
        
        # Select best overall scenario (highest overall score)
        best_scenario = max(pareto_scenarios, key=lambda x: x['metrics']['overall_score'])
        
        # Load raw project data for frontend
        with open(file_path, 'r') as f:
            project_data = json.load(f)
        
        return {
            "success": True,
            "best_scenario": best_scenario,
            "project_data": project_data,
            "constraints": extract_constraints(project_data),
            "all_scenarios": pareto_scenarios
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

def extract_constraints(project_data: Dict) -> Dict:
    """Extract current constraints from project data"""
    return {
        "resources": {
            resource["id"]: {
                "hourly_rate": resource["hourly_rate"],
                "max_hours_per_day": resource["max_hours_per_day"],
                "available": True,
                "skills": resource["skills"]
            }
            for resource in project_data["resources"]
        },
        "tasks": {
            task["id"]: {
                "duration_hours": task["duration_hours"],
                "priority": 3,  # Default normal priority
                "allow_parallel": False,
                "required_skills": task["required_skills"],
                "order": task["order"]
            }
            for task in project_data["tasks"]
        },
        "preferences": {
            "time_priority": 0.33,
            "cost_priority": 0.33,
            "quality_priority": 0.34
        }
    }

def apply_constraints(project_data, constraints):
    """Apply custom constraints to project data"""
    # Apply resource constraints
    if constraints.resources:
        for resource_id, resource_constraint in constraints.resources.items():
            for resource in project_data.get('resources', []):
                if resource['id'] == resource_id:
                    if hasattr(resource_constraint, 'hourly_rate'):
                        resource['hourly_rate'] = resource_constraint.hourly_rate
                    if hasattr(resource_constraint, 'max_hours_per_day'):
                        resource['max_hours_per_day'] = resource_constraint.max_hours_per_day
                    if hasattr(resource_constraint, 'available'):
                        resource['available'] = resource_constraint.available
    
    # Apply task constraints
    if constraints.tasks:
        for task_id, task_constraint in constraints.tasks.items():
            for task in project_data.get('tasks', []):
                if task['id'] == task_id:
                    if hasattr(task_constraint, 'duration_hours'):
                        task['duration_hours'] = task_constraint.duration_hours
                    if hasattr(task_constraint, 'priority'):
                        task['priority'] = task_constraint.priority
    
    return project_data

def apply_direct_constraints(project_data, constraint_data):
    """Apply constraints from direct dictionary format"""
    # Apply resource constraints
    resources = constraint_data.get('resources', {})
    for resource_id, resource_constraint in resources.items():
        for resource in project_data.get('resources', []):
            if resource['id'] == resource_id:
                if 'hourly_rate' in resource_constraint:
                    resource['hourly_rate'] = resource_constraint['hourly_rate']
                if 'max_hours_per_day' in resource_constraint:
                    resource['max_hours_per_day'] = resource_constraint['max_hours_per_day']
                if 'available' in resource_constraint:
                    resource['available'] = resource_constraint['available']
    
    # Apply task constraints
    tasks = constraint_data.get('tasks', {})
    for task_id, task_constraint in tasks.items():
        for task in project_data.get('tasks', []):
            if task['id'] == task_id:
                if 'duration_hours' in task_constraint:
                    task['duration_hours'] = task_constraint['duration_hours']
                if 'priority' in task_constraint:
                    task['priority'] = task_constraint['priority']
    
    return project_data

async def get_cms_processes():
    """Fetch processes from CMS API"""
    try:
        # Authenticate with CMS
        async with httpx.AsyncClient(timeout=10.0) as client:
            auth_response = await client.post(
                "https://server-digitaltwin-enterprise-production.up.railway.app/auth/login",
                json={
                    "email": "superadmin@example.com",
                    "password": "ChangeMe123!"
                }
            )
            
            if auth_response.status_code in [200, 201]:
                auth_data = auth_response.json()
                access_token = auth_data['access_token']
                
                # Fetch processes
                processes_response = await client.get(
                    "https://server-digitaltwin-enterprise-production.up.railway.app/process/with-relations",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if processes_response.status_code == 200:
                    return processes_response.json()
                    
    except Exception as e:
        print(f"CMS API error: {e}")
        return None
    
    return None

async def get_cms_process_by_id(process_id: int):
    """Fetch a specific process from CMS API by ID"""
    try:
        # Authenticate with CMS
        async with httpx.AsyncClient(timeout=30.0) as client:
            auth_response = await client.post(
                "https://server-digitaltwin-enterprise-production.up.railway.app/auth/login",
                json={
                    "email": "superadmin@example.com",
                    "password": "ChangeMe123!"
                }
            )
            
            if auth_response.status_code in [200, 201]:
                auth_data = auth_response.json()
                access_token = auth_data['access_token']
                
                # Fetch specific process
                process_response = await client.get(
                    f"https://server-digitaltwin-enterprise-production.up.railway.app/process/{process_id}/with-relations",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if process_response.status_code == 200:
                    return process_response.json()
                else:
                    print(f"Failed to fetch process {process_id}: {process_response.status_code} - {process_response.text}")
                    return None
            else:
                print(f"Authentication failed: {auth_response.status_code} - {auth_response.text}")
                return None
                    
    except Exception as e:
        print(f"CMS API error for process {process_id}: {e}")
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

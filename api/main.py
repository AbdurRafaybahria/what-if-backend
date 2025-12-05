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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.data_models import Project
from src.models.cms_transformer import transform_cms_to_internal, validate_cms_data, get_cms_transformation_summary
from src.optimization.scenario_generator import ScenarioGenerator
from src.optimization.pareto_optimizer import ParetoOptimizer

app = FastAPI(title="What-If Analysis API", version="1.0.0")

# Load CMS configuration from environment variables
CMS_BASE_URL = os.getenv("CMS_BASE_URL")
CMS_EMAIL = os.getenv("CMS_EMAIL")
CMS_PASSWORD = os.getenv("CMS_PASSWORD")

# Validate required environment variables
if not all([CMS_BASE_URL, CMS_EMAIL, CMS_PASSWORD]):
    raise ValueError("Missing required environment variables: CMS_BASE_URL, CMS_EMAIL, CMS_PASSWORD")

# Enable CORS for frontend with specific HTTPS origins and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fyp-cms-frontend.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Alternative dev port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://192.168.100.15:3000",
        "*"  # Allow all origins as fallback
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    expose_headers=["*"]
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

@app.get("/optimize/cms-process/{process_id}")
async def optimize_cms_process_by_id(process_id: int):
    """Fetch CMS process by ID and optimize it"""
    try:
        # Fetch process data from CMS API
        cms_data = await get_cms_process_by_id(process_id)
        if not cms_data:
            raise HTTPException(status_code=404, detail=f"Process with ID {process_id} not found")
        
        # Debug: Print what we received from CMS
        print(f"DEBUG CMS Response keys: {list(cms_data.keys()) if isinstance(cms_data, dict) else type(cms_data)}")
        
        # Validate CMS data structure
        if not validate_cms_data(cms_data):
            # Debug: Identify what's missing
            missing = []
            for field in ['process_id', 'process_name', 'process_overview']:
                if field not in cms_data:
                    missing.append(field)
            process_tasks = cms_data.get('process_task', cms_data.get('process_tasks', []))
            print(f"DEBUG Missing fields: {missing}, process_tasks count: {len(process_tasks) if isinstance(process_tasks, list) else 'NOT A LIST'}")
            if process_tasks and len(process_tasks) > 0:
                print(f"DEBUG First process_task keys: {list(process_tasks[0].keys()) if isinstance(process_tasks[0], dict) else 'NOT A DICT'}")
            raise HTTPException(status_code=400, detail=f"Invalid CMS data format. Missing: {missing}")
        
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
        
        # Extract task and resource names for response
        task_names = {}
        resource_names = {}
        
        process_tasks = cms_data.get('process_task', cms_data.get('process_tasks', []))
        for process_task in process_tasks:
            task_data = process_task['task']
            task_id = f"task_{task_data['task_id']:03d}"
            task_names[task_id] = task_data['task_name']
            
            for job_task in task_data['jobTasks']:
                job = job_task['job']
                resource_id = f"resource_{job['job_id']:03d}"
                resource_names[resource_id] = job['name']
        
        return {
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "baseline": cms_baseline.to_dict(),
            "process_info": {
                "process_id": cms_data['process_id'],
                "process_name": cms_data['process_name'],
                "company": cms_data.get('company', {}).get('name', 'Unknown')
            },
            "task_names": task_names,
            "resource_names": resource_names
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
    """Fetch processes from CMS API using HttpOnly cookie authentication"""
    try:
        # Use httpx.AsyncClient with cookies enabled for HttpOnly cookie auth
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Authenticate with CMS - cookie will be set automatically in response
            auth_response = await client.post(
                f"{CMS_BASE_URL}/auth/login",
                json={
                    "email": CMS_EMAIL,
                    "password": CMS_PASSWORD
                }
            )
            
            if auth_response.status_code in [200, 201]:
                # Extract cookies from login response
                cookies = auth_response.cookies
                
                # Fetch processes - pass cookies instead of Authorization header
                processes_response = await client.get(
                    f"{CMS_BASE_URL}/process/with-relations",
                    cookies=cookies
                )
                
                if processes_response.status_code == 200:
                    return processes_response.json()
                else:
                    print(f"Failed to fetch processes: {processes_response.status_code} - {processes_response.text}")
                    
    except Exception as e:
        print(f"CMS API error: {e}")
        return None
    
    return None

async def get_cms_process_by_id(process_id: int):
    """Fetch a specific process from CMS API by ID using HttpOnly cookie authentication"""
    try:
        # Use httpx.AsyncClient with cookies enabled for HttpOnly cookie auth
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Authenticate with CMS - cookie will be set automatically in response
            auth_response = await client.post(
                f"{CMS_BASE_URL}/auth/login",
                json={
                    "email": CMS_EMAIL,
                    "password": CMS_PASSWORD
                }
            )
            
            print(f"DEBUG Auth response status: {auth_response.status_code}")
            print(f"DEBUG Auth response cookies: {list(auth_response.cookies.keys())}")
            
            if auth_response.status_code in [200, 201]:
                # Extract cookies from login response
                cookies = auth_response.cookies
                
                # Fetch specific process - pass cookies instead of Authorization header
                process_response = await client.get(
                    f"{CMS_BASE_URL}/process/{process_id}/with-relations",
                    cookies=cookies
                )
                
                print(f"DEBUG Process response status: {process_response.status_code}")
                
                if process_response.status_code == 200:
                    data = process_response.json()
                    print(f"DEBUG Process response type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                    return data
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
    import os
    port = int(os.environ.get("PORT", 8002))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # For development, allow HTTP from HTTPS origins
    # In production, you should use proper SSL certificates
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        # ssl_keyfile="path/to/key.pem",  # Uncomment for HTTPS
        # ssl_certfile="path/to/cert.pem"  # Uncomment for HTTPS
    )
